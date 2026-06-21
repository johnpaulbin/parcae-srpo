"""
Recurrent-Depth Gemma 4 12B; retrofit a pretrained 48-layer model
into a middle-looped recurrent-depth architecture (Parcae, 2026).

Architecture:
    Prelude P:   layers 0--15  (16 of 48 layers, run once)
    Recurrent R: layers 16--31 (16 of 48 layers, looped T times with injection)
    Coda C:      layers 32--47 (16 of 48 layers, run once)

Injection follows the Parcae stable update rule:
    h_0 = P(x)
    h_{t+1} = A·h_t + B·e + R(h_t + e)    for t = 0..T-1
    output = C(h_T)

where:
    e = h_0 (encoded input frozen across loops; prevents drift)
    A, B = LTI-stable injection parameters (guaranteed ρ(A) < 1)

Layer pattern for Gemma 4 12B (48 layers, 8 blocks of [5s+1g]):
    Block 0: layers 0--5   → Prelude start
    Block 1: layers 6--11  → Prelude end
    Block 2: layers 12--17 → Recurrent start
    Block 3: layers 18--23 → Recurrent mid
    Block 4: layers 24--29 → Recurrent mid
    Block 5: layers 30--35 → Recurrent end
    Block 6: layers 36--41 → Coda start
    Block 7: layers 42--47 → Coda end

With T=1: identical to the original 48-layer model (baseline check)
With T=3: effective depth = 16 + 16×3 + 16 = 80 layers
With T=6: effective depth = 16 + 16×6 + 16 = 128 layers
"""

from dataclasses import dataclass
from typing import Optional
import collections
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModelForCausalLM

from .injection import LTIInjection


def _resolve_module_path(root, path: str):
    """Resolve a dotted module path, returning None if any segment is missing."""
    node = root
    for part in path.split("."):
        if not hasattr(node, part):
            return None
        node = getattr(node, part)
    return node


def _find_first_module_path(root, paths: list[str]):
    """Return the first existing module path and value from a wrapper object."""
    for path in paths:
        value = _resolve_module_path(root, path)
        if value is not None:
            return path, value
    return None, None


def _find_gemma4_mask_fns(language_model):
    """Resolve Gemma 4 mask helpers from the loaded model implementation."""
    import importlib

    module = importlib.import_module(language_model.__class__.__module__)
    return (
        getattr(module, "create_causal_mask", None),
        getattr(module, "create_sliding_window_causal_mask", None),
    )


def _layer_forward(layer, hidden_states, per_layer_input, shared_kv_states,
                   position_embeddings, attention_mask, position_ids,
                   use_cache=False):
    """Standalone function for torch.utils.checkpoint (must be picklable)."""
    layer_out = layer(
        hidden_states=hidden_states,
        per_layer_input=per_layer_input,
        shared_kv_states=shared_kv_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=use_cache,
    )
    if use_cache and isinstance(layer_out, tuple):
        attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
        if attn is not None:
            attn._last_present_kv = layer_out[1]
        return layer_out[0]
    return layer_out


@dataclass
class RecurrentDepthConfig:
    """Configuration for the recurrent-depth retrofit."""

    # Path to the pretrained Gemma 4 model
    model_path: str = "google/gemma-4-E2B-it"
    load_backend: str = "transformers"     # transformers | unsloth
    load_in_4bit: bool = False             # used by the Unsloth backend
    max_seq_length: int = 2048             # Unsloth rope/cache setup length
    fast_inference: bool = False           # leave False for training memory

    # Layer split must sum to model's total (e.g., 12+11+12=35 for E2B)
    prelude_layers: int = 12
    n_recurrent_layers: int = 11  # these get looped
    coda_layers: int = 12

    # Default loop count at inference (T in the equations)
    default_loops: int = 1

    # Whether to add depth-wise LoRA adapters in the recurrent block
    # (per-loop parameter variation, as in OpenMythos / LoopFormer)
    use_depth_lora: bool = True
    lora_rank: int = 16

    # Whether to add a sinusoidal loop-index embedding to the hidden state
    # at each iteration (distinguishes early from late loop passes)
    use_loop_embedding: bool = True
    loop_embedding_dim: int = 256  # channels receiving the loop signal

    # Recompute frozen backbone activations during backward. This is essential
    # when the recurrent block is looped several times on a single A100.
    use_activation_checkpointing: bool = True


class LoopEmbedding(nn.Module):
    """
    Sinusoidal loop-index embedding injected into the first N channels of h.

    Without this, the same recurrent block weights are applied identically at
    every iteration. Adding a loop-t position signal lets the same weights
    implement functionally distinct operations at different depths.

    Analogous to positional embeddings but over recurrence depth, not sequence
    position. Uses sine/cosine frequencies as in standard sinusoidal PE.
    """

    def __init__(self, dim: int, loop_dim: int, max_loops: int = 64):
        super().__init__()
        self.loop_dim = min(loop_dim, dim)
        self.max_loops = max_loops
        # Precompute the embedding table for fast lookup
        freqs = 1.0 / (
            10000.0 ** (torch.arange(0, self.loop_dim, 2).float() / self.loop_dim)
        )
        positions = torch.arange(max_loops).float()
        angles = torch.outer(positions, freqs)  # (max_loops, loop_dim//2)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        self.register_buffer("embedding_table", emb[:, : self.loop_dim])

    def forward(self, h: torch.Tensor, loop_t: int) -> torch.Tensor:
        """Add loop-index signal to first loop_dim channels of h."""
        t = min(loop_t, self.max_loops - 1)
        emb = self.embedding_table[t]  # (loop_dim,)
        out = h + 0  # new tensor, no in-place mutation of caller's h
        out[..., : self.loop_dim] = out[..., : self.loop_dim] + emb
        return out


class DepthLoRA(nn.Module):
    """
    Per-loop LoRA adapter for the recurrent block.

    Same base architecture as standard LoRA but with a per-loop scale factor
    that modulates the low-rank adapter output. This lets the same base weights
    produce subtly different behavior at each loop iteration without adding
    significant parameters.

    delta(x, t) = scale[t] * (x @ A) @ B

    where A ∈ R^{dim × rank} is the down-projection (shared across loops),
    B ∈ R^{rank × dim} is the up-projection (shared across loops),
    and scale[t] ∈ R^{rank} is a per-loop element-wise modulation.
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = nn.Embedding(max_loops, rank)
        # Initialize near-zero so the adapter starts with negligible effect
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.scale.weight, std=0.01)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """Apply per-loop LoRA delta to x."""
        max_t = self.scale.num_embeddings - 1
        t_idx = min(loop_t, max_t)
        s = self.scale(torch.tensor(t_idx, device=x.device)).to(x.dtype)  # (rank,)
        return self.up(self.down(x.to(self.down.weight.dtype)) * s).to(x.dtype)  # (B, T, dim)


class RecurrentDepthGemma(nn.Module):
    """
    Gemma 4 retrofitted as a recurrent-depth transformer.

    Loads the full pretrained HuggingFace model, then restructures the
    layers into Prelude → Recurrent → Coda according to the split config.
    Adds LTIInjection, optional depth-wise LoRA, and loop-index embeddings.

    Usage:
        cfg = RecurrentDepthConfig(model_path="google/gemma-4-E2B-it",
                                    prelude_layers=12,
                                    n_recurrent_layers=11,
                                    coda_layers=12)
        model = RecurrentDepthGemma(cfg)
        model.load_pretrained()
        # Now model behaves like the original at n_loops=1

    Training:
        The injection parameters (A, B) and depth-wise LoRA weights are
        trainable. The original Gemma weights are frozen in bf16.
        Use variable T sampling (Poisson mean μ_rec) during training,
        per the Parcae training recipe.
    """

    def __init__(self, cfg: RecurrentDepthConfig):
        super().__init__()
        self.cfg = cfg

        # Load config from the pretrained model
        try:
            hf_config = AutoConfig.from_pretrained(
                cfg.model_path,
                trust_remote_code=True,
            )
        except ValueError as exc:
            msg = str(exc)
            if "does not recognize this architecture" in msg or "model_type" in msg:
                raise ValueError(
                    f"Could not load config for {cfg.model_path!r}: the installed "
                    "Transformers has no native support for this architecture. "
                    "Install a Transformers version with native support. If using "
                    "the Unsloth backend, keep Transformers <= 4.57.2; otherwise "
                    "switch to load_backend='transformers' and upgrade Transformers."
                ) from exc
            raise
        self.hidden_size = hf_config.text_config.hidden_size
        self.num_layers = hf_config.text_config.num_hidden_layers

        # Validate split
        total = cfg.prelude_layers + cfg.n_recurrent_layers + cfg.coda_layers
        if total != self.num_layers:
            raise ValueError(
                f"Layer split {cfg.prelude_layers}+{cfg.n_recurrent_layers}"
                f"+{cfg.coda_layers} = {total}, but model has {self.num_layers} layers. "
                f"Split must sum to {self.num_layers}."
            )

        # Create the full model structure. We'll populate weights from pretrained.
        # Using a lazy approach: load the full HF model, extract layers, reassign.
        self._hf_config = hf_config
        self._layers: dict[int, nn.Module] = {}

        # --- New modules (not from pretrained) ---
        self.injection = LTIInjection(self.hidden_size)
        self.intermediate_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

        if cfg.use_loop_embedding:
            self.loop_embed = LoopEmbedding(
                self.hidden_size,
                cfg.loop_embedding_dim,
                max_loops=64,
            )
        else:
            self.loop_embed = None

        if cfg.use_depth_lora:
            self.depth_lora = DepthLoRA(
                self.hidden_size,
                cfg.lora_rank,
                max_loops=64,
            )
        else:
            self.depth_lora = None

        # Layer indices for each block
        self.prelude_indices = list(range(0, cfg.prelude_layers))
        self.recurrent_indices = list(
            range(cfg.prelude_layers, cfg.prelude_layers + cfg.n_recurrent_layers)
        )
        self.coda_indices = list(
            range(
                cfg.prelude_layers + cfg.n_recurrent_layers,
                cfg.prelude_layers + cfg.n_recurrent_layers + cfg.coda_layers,
            )
        )

        # Will be populated by load_pretrained()
        self._is_loaded = False

    def load_pretrained(self):
        """
        Load the full HF Gemma 4 model and extract its layers and non-layer modules.

        This loads the 23GB safetensors file; requires ~24GB RAM or memory-mapped
        loading. The model is NOT kept in GPU memory; layers are stored on CPU
        and moved to device during forward pass.

        After loading, the HF model object is deleted to free memory.
        """
        print(f"Loading pretrained model from {self.cfg.model_path}...")
        backend = self.cfg.load_backend.lower()
        def _load_transformers(load_in_4bit: bool):
            kwargs = {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
            else:
                kwargs["device_map"] = "cpu"
            return AutoModelForCausalLM.from_pretrained(self.cfg.model_path, **kwargs)

        if backend == "transformers":
            hf_model = _load_transformers(self.cfg.load_in_4bit)
            self._quantized_backbone = bool(self.cfg.load_in_4bit)
        elif backend == "unsloth":
            try:
                from unsloth import FastLanguageModel
            except Exception as exc:
                if not self.cfg.load_in_4bit:
                    raise ImportError(
                        "The Unsloth backend requires a working `unsloth` "
                        "installation."
                    ) from exc
                warnings.warn(
                    "Unsloth import failed; falling back to Transformers "
                    f"bitsandbytes 4-bit loading. Original error: {exc}",
                    RuntimeWarning,
                )
                hf_model = _load_transformers(load_in_4bit=True)
                backend = "transformers-bnb"
                self._quantized_backbone = True
                FastLanguageModel = None
            unsloth_kwargs = {
                "model_name": self.cfg.model_path,
                "max_seq_length": self.cfg.max_seq_length,
                "dtype": None,
                "load_in_4bit": self.cfg.load_in_4bit,
                "fast_inference": self.cfg.fast_inference,
                "trust_remote_code": True,
            }
            if FastLanguageModel is not None:
                try:
                    try:
                        hf_model, _tokenizer = FastLanguageModel.from_pretrained(**unsloth_kwargs)
                    except TypeError as exc:
                        if "trust_remote_code" not in str(exc):
                            raise
                        unsloth_kwargs.pop("trust_remote_code", None)
                        hf_model, _tokenizer = FastLanguageModel.from_pretrained(**unsloth_kwargs)
                    self._quantized_backbone = bool(self.cfg.load_in_4bit)
                except Exception as exc:
                    if not self.cfg.load_in_4bit:
                        raise
                    warnings.warn(
                        "Unsloth loading failed; falling back to Transformers "
                        f"bitsandbytes 4-bit loading. Original error: {exc}",
                        RuntimeWarning,
                    )
                    hf_model = _load_transformers(load_in_4bit=True)
                    backend = "transformers-bnb"
                    self._quantized_backbone = True
        else:
            raise ValueError(
                f"Unknown load_backend={self.cfg.load_backend!r}; "
                "expected 'transformers' or 'unsloth'."
            )

        # Gemma 4 is nested differently by raw HF, PEFT, and Unsloth wrappers.
        lm_path, language_model = _find_first_module_path(
            hf_model,
            [
                "model.language_model",
                "model.model.language_model",
                "base_model.model.model.language_model",
                "base_model.model.language_model",
                "base_model.model.model.model.language_model",
            ],
        )
        if language_model is None:
            base_getter = getattr(hf_model, "get_base_model", None)
            if callable(base_getter):
                base_model = base_getter()
                lm_path, language_model = _find_first_module_path(
                    base_model,
                    [
                        "model.language_model",
                        "model.model.language_model",
                        "language_model",
                    ],
                )
        if language_model is None:
            raise AttributeError(
                "Could not locate the Gemma 4 language_model inside the "
                f"{self.cfg.load_backend} model wrapper."
            )
        self._create_causal_mask, self._create_sliding_window_causal_mask = (
            _find_gemma4_mask_fns(language_model)
        )

        # Extract layers
        for idx in range(self.num_layers):
            self._layers[idx] = language_model.layers[idx]

        # Extract non-layer components
        self.embed_tokens = language_model.embed_tokens
        self.norm = language_model.norm
        _head_path, lm_head = _find_first_module_path(
            hf_model,
            [
                "lm_head",
                "base_model.model.lm_head",
                "base_model.model.model.lm_head",
            ],
        )
        if lm_head is None:
            raise AttributeError("Could not locate lm_head on the loaded model.")
        self.lm_head = lm_head
        self.rotary_emb = language_model.rotary_emb
        self._language_model = language_model  # keep for PLE methods

        # Vision/audio embedders; keep for multimodal inputs
        _container_path, model_container = _find_first_module_path(
            hf_model,
            [
                "model",
                "base_model.model.model",
                "base_model.model",
            ],
        )
        self.embed_vision = getattr(model_container, "embed_vision", None)
        self.embed_audio = getattr(model_container, "embed_audio", None)

        # Move all layers to CPU (they were loaded there)
        # The trainer will call model.to(device) to move them to GPU
        # But _layers is a plain dict, not nn.ModuleDict, so we need
        # to register them as children for .to() to work.
        self._layer_module_list = torch.nn.ModuleList(
            [self._layers[i] for i in range(self.num_layers)]
        )
        # Check if model uses Per-Layer Embeddings (E2B, E4B)
        ple_dim = getattr(language_model.config, "hidden_size_per_layer_input", 0)
        self._ple_enabled = ple_dim > 0

        # Final logit softcapping (Gemma 4 applies tanh scaling after lm_head)
        self._logit_softcap = getattr(
            language_model.config, 'final_logit_softcapping', None)

        # EOS token id for generation
        self._eos_id = getattr(
            getattr(language_model, 'config', None), 'eos_token_id', 1)

        self._is_loaded = True

        # Free the HF wrapper; we have all the pieces
        del hf_model
        print(
            f"Loaded {self.num_layers} layers. "
            f"Prelude: {len(self.prelude_indices)}, "
            f"Recurrent: {len(self.recurrent_indices)}, "
            f"Coda: {len(self.coda_indices)}"
        )
        if backend == "unsloth":
            print(
                f"Unsloth backend active. language_model={lm_path}, "
                f"load_in_4bit={self.cfg.load_in_4bit}"
            )
        print(f"Injection rho(A) = {self.injection.compute_spectral_radius():.6f}")

    @property
    def quantized_backbone(self) -> bool:
        return bool(getattr(self, "_quantized_backbone", False))

    def move_trainable_modules(self, device: torch.device | str):
        """Move new SRPO modules when a quantized backbone cannot use .to()."""
        self.injection.to(device)
        self.intermediate_norm.to(device)
        if self.loop_embed is not None:
            self.loop_embed.to(device)
        if self.depth_lora is not None:
            self.depth_lora.to(device)

    def _compute_ple(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute Per-Layer Embeddings for E2B/E4B models.
        
        Returns tensor of shape (B, T, num_layers, ple_dim) or None if PLE disabled.
        The per-layer input for layer i is per_layer_inputs[:, :, i, :].
        PLE methods live on the language model, not on embed_tokens.
        """
        if not self._ple_enabled:
            return None
        lm = self._language_model
        # token-identity component: lookup input_ids in per-layer embedding table
        ple_token = lm.get_per_layer_inputs(input_ids, inputs_embeds)
        # context-aware component: project hidden states + combine
        ple_full = lm.project_per_layer_inputs(inputs_embeds, ple_token)
        return ple_full  # (B, T, num_layers, ple_dim)

    def _compute_position_embeddings(
        self, h: torch.Tensor, position_ids: torch.Tensor
    ) -> dict:
        """Precompute RoPE cos/sin for each attention type, cast to h.dtype."""
        pos_embeds = {}
        for layer_type in ["sliding_attention", "full_attention"]:
            cos, sin = self.rotary_emb(h, position_ids, layer_type)
            pos_embeds[layer_type] = (cos.to(h.dtype), sin.to(h.dtype))
        return pos_embeds

    def _run_block(
        self,
        h: torch.Tensor,
        layer_indices: list[int],
        position_embeddings: dict,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        shared_kv_states: Optional[dict] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
        return_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Run a contiguous block of Gemma 4 decoder layers.

        Each layer receives the correct (cos, sin) tuple from position_embeddings
        keyed by its attention type (sliding_attention or full_attention).

        If use_checkpoint=True, each layer is wrapped in torch.utils.checkpoint
        to trade compute for memory; activations are recomputed during backward.

        Returns:
            If return_kv_cache=False: hidden states (B, T, dim)
            If return_kv_cache=True:  (hidden_states, {layer_idx: present_kv})
        """
        if shared_kv_states is None:
            shared_kv_states = collections.UserDict()
        # shared_kv_states persists across blocks within one forward pass.
        # Some layers store full-length KV here; later shared layers read it.
        # Do NOT clear between loop iterations — KV sharing is per-pass.
        for idx in layer_indices:
            layer = self._layer_module_list[idx]
            layer_type = self._hf_config.text_config.layer_types[idx]
            # Get per-layer input for this specific layer index
            ple = None
            if per_layer_inputs is not None:
                ple = per_layer_inputs[:, :, idx, :]  # (B, T, ple_dim)
            # Look up attention mask for this layer type
            layer_mask = attention_mask[layer_type] if isinstance(attention_mask, dict) else attention_mask

            if use_checkpoint:
                h = torch.utils.checkpoint.checkpoint(
                    _layer_forward, layer, h, ple, shared_kv_states,
                    position_embeddings[layer_type], layer_mask, position_ids,
                    return_kv_cache,
                    use_reentrant=False,
                )
            else:
                layer_out = layer(
                    hidden_states=h,
                    per_layer_input=ple,
                    shared_kv_states=shared_kv_states,
                    position_embeddings=position_embeddings[layer_type],
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    use_cache=return_kv_cache,
                )
                # HF layer returns (h,) or (h, present_kv) if use_cache=True
                if return_kv_cache and isinstance(layer_out, tuple):
                    h = layer_out[0]
                    # Store KV on the attention module for _steal_kv_cache
                    attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
                    if attn is not None:
                        attn._last_present_kv = layer_out[1]
                else:
                    h = layer_out if not isinstance(layer_out, tuple) else layer_out[0]
        if return_kv_cache:
            kv_out = {}
            for idx in layer_indices:
                layer = self._layer_module_list[idx]
                attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
                if attn is not None and hasattr(attn, '_last_present_kv'):
                    kv_out[idx] = attn._last_present_kv
            return h, kv_out
        return h

    def _recurrent_loop(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        n_loops: int,
        run_block_fn,
        return_kv_cache: bool = False,
        show_work: bool = False,
    ):
        """Core recurrent loop shared by forward() and generate().

        At each iteration t:
          1. BPTT detach (forward-only, via _bptt_depth attribute)
          2. Loop-index embedding (skip t=0)
          3. run_block_fn(h, t) returns transformer output (+ optional KV)
          4. Depth-wise LoRA adapter
          5. Injection: bypass at t=0; (A-I)·h + B·e + trans_out at t>=1
          6. If show_work: project h to logits via intermediate_norm + lm_head

        Args:
            h: hidden state entering the loop
            e: frozen prelude output for injection
            n_loops: number of iterations
            run_block_fn: callable (h, t) -> trans_out or (trans_out, kv)
            return_kv_cache: run_block_fn returns KV tuples
            show_work: capture per-loop intermediate logits. Inference
                       only -- raises RuntimeError if grad is enabled.

        Returns:
            Always returns (h, kv, thoughts) tuple.
            kv is None unless return_kv_cache=True.
            thoughts is None unless show_work=True (then list of
            (loop_t, logits)).
        """
        if show_work and torch.is_grad_enabled():
            raise RuntimeError(
                "show_work is inference-only. Disable grad or set "
                "show_work=False for training."
            )

        bptt_depth = getattr(self, '_bptt_depth', None)
        detach_before = None
        if bptt_depth is not None and bptt_depth < n_loops:
            detach_before = n_loops - bptt_depth

        kv_rec = [] if return_kv_cache else None
        thoughts = [] if show_work else None

        for t in range(n_loops):
            if detach_before is not None and t < detach_before:
                h = h.detach()

            if self.loop_embed is not None and t > 0:
                h = self.loop_embed(h, t)

            rec_out = run_block_fn(h, t)
            if return_kv_cache and isinstance(rec_out, tuple):
                trans_out, kv_t = rec_out
                kv_rec.append(kv_t)
            else:
                trans_out = rec_out

            if self.depth_lora is not None:
                trans_out = trans_out + self.depth_lora(trans_out, t)

            if t == 0:
                h = trans_out
            else:
                h = self.injection(h, e, trans_out)

            if show_work:
                h_norm = self.intermediate_norm(h)
                inter_logits = self.lm_head(h_norm)
                if self._logit_softcap is not None:
                    inter_logits = self._logit_softcap * torch.tanh(
                        inter_logits.float() / self._logit_softcap
                    ).to(inter_logits.dtype)
                thoughts.append((t, inter_logits.detach()))

        return h, kv_rec, thoughts

    def _forward_pipeline(
        self,
        input_ids: torch.Tensor,
        n_loops: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_kv_cache: bool = False,
        show_work: bool = False,
    ):
        """Prelude -> Recurrent -> Coda, returning (h, kv_rec, thoughts).

        Shared by forward(), forward_with_thoughts(), and generate().
        h is the hidden state after coda (before final norm + lm_head).
        kv_rec is None unless return_kv_cache=True.
        thoughts is None unless show_work=True.
        """
        B, T = input_ids.shape
        device = input_ids.device

        h = self.embed_tokens(input_ids)
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        else:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_embeds = self._compute_position_embeddings(h, position_ids)
        if self._create_causal_mask is None or self._create_sliding_window_causal_mask is None:
            raise RuntimeError(
                "The loaded Gemma 4 implementation does not expose causal mask "
                "helpers. Upgrade Transformers to a Gemma-4-capable build or "
                "use a model repository with trusted remote modeling code."
            )
        causal_mask_mapping = {
            "full_attention": self._create_causal_mask(
                config=self._hf_config.text_config, inputs_embeds=h,
                attention_mask=attention_mask, past_key_values=None,
                position_ids=position_ids),
            "sliding_attention": self._create_sliding_window_causal_mask(
                config=self._hf_config.text_config, inputs_embeds=h,
                attention_mask=attention_mask, past_key_values=None,
                position_ids=position_ids),
        }
        shared_kv_states = collections.UserDict()
        per_layer_inputs = None
        if hasattr(self, "_ple_enabled") and self._ple_enabled:
            per_layer_inputs = self._compute_ple(input_ids, h)
        use_checkpoint = (
            self.cfg.use_activation_checkpointing
            and torch.is_grad_enabled()
            and not return_kv_cache
        )

        prelude_out = self._run_block(
            h, self.prelude_indices, pos_embeds, causal_mask_mapping,
            position_ids, shared_kv_states=shared_kv_states,
            per_layer_inputs=per_layer_inputs,
            use_checkpoint=use_checkpoint,
            return_kv_cache=return_kv_cache)
        h = prelude_out[0] if isinstance(prelude_out, tuple) else prelude_out
        e = h.clone().detach()

        def _run_rec(h, _t):
            return self._run_block(
                h, self.recurrent_indices, pos_embeds,
                causal_mask_mapping, position_ids,
                shared_kv_states=shared_kv_states,
                per_layer_inputs=per_layer_inputs,
                use_checkpoint=use_checkpoint,
                return_kv_cache=return_kv_cache)

        h, kv_rec, thoughts = self._recurrent_loop(
            h, e, n_loops, _run_rec,
            return_kv_cache=return_kv_cache,
            show_work=show_work)

        coda_out = self._run_block(
            h, self.coda_indices, pos_embeds, causal_mask_mapping,
            position_ids, shared_kv_states=shared_kv_states,
            per_layer_inputs=per_layer_inputs,
            use_checkpoint=use_checkpoint,
            return_kv_cache=return_kv_cache)
        h = coda_out[0] if isinstance(coda_out, tuple) else coda_out
        return h, kv_rec, thoughts

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        return_kv_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass: Prelude → Recurrent (looped) → Coda → lm_head.

        Args:
            input_ids: Token indices of shape (B, T)
            n_loops: Number of recurrent loop iterations.
                     Defaults to cfg.default_loops.
                     Set to 1 to match the original pretrained baseline.
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            return_logits: If True, return logits. If False, return hidden states.
            return_kv_cache: Return KV caches for incremental generation.

        Returns:
            Logits (B, T, V) or hidden states (B, T, D).
        """
        if not self._is_loaded:
            raise RuntimeError("Call load_pretrained() before forward().")

        n_loops = n_loops if n_loops is not None else self.cfg.default_loops
        if n_loops < 1:
            raise ValueError(f"n_loops must be >= 1 (got {n_loops}). 1 = identity.")
        self._last_n_loops = n_loops

        h, kv_rec, _ = self._forward_pipeline(
            input_ids, n_loops,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_kv_cache=return_kv_cache,
            show_work=False,
        )
        if return_kv_cache:
            self._kv_rec = kv_rec

        if not return_logits:
            return h

        h = self.norm(h)
        logits = self.lm_head(h)
        if self._logit_softcap is not None:
            logits = self._logit_softcap * torch.tanh(logits.float() / self._logit_softcap).to(logits.dtype)
        return logits

    def forward_with_thoughts(
        self,
        input_ids: torch.Tensor,
        n_loops: int = 2,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        """Forward pass returning per-loop intermediate logits.

        Projects the hidden state to logits after each recurrent loop
        iteration using intermediate_norm + lm_head.  Useful for
        inspecting how the model's predictions evolve with depth.

        intermediate_norm is initialized to identity (weight=1, bias=0)
        and is NOT pretrained; intermediate logits are approximate.
        Inference-only; raises RuntimeError if grad is enabled.

        Returns:
            (final_logits, thoughts) where thoughts is
            [(loop_t, inter_logits), ...].
        """
        if not self._is_loaded:
            raise RuntimeError("Call load_pretrained() first.")
        if n_loops < 1:
            raise ValueError(f"n_loops must be >= 1 (got {n_loops}).")
        self._last_n_loops = n_loops

        h, _, thoughts = self._forward_pipeline(
            input_ids, n_loops,
            attention_mask=attention_mask,
            position_ids=position_ids,
            show_work=True)

        h = self.norm(h)
        logits = self.lm_head(h)
        if self._logit_softcap is not None:
            logits = self._logit_softcap * torch.tanh(
                logits.float() / self._logit_softcap).to(logits.dtype)
        return logits, thoughts

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 1,
        temperature: float = 0.7,
        top_k: int = 50,
        return_logprobs: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive generation with full KV caching.

        If return_logprobs=True, returns (token_ids, token_logprobs) where
        token_logprobs contains the log-prob of each generated token under
        the raw (pre-sampling) distribution.  Single forward pass.
        """
        device = input_ids.device
        B, prompt_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)

        prompt_lengths = attention_mask.long().sum(dim=-1)
        if not torch.equal(prompt_lengths, torch.full_like(prompt_lengths, prompt_len)):
            raise ValueError(
                "RecurrentDepthGemma.generate expects unpadded input_ids. "
                "Generate one prompt at a time, or pass sequences with equal "
                "non-padding length."
            )

        # Token 0: full forward + capture KV
        logits_full = self.forward(
            input_ids, n_loops=n_loops, return_logits=True,
            return_kv_cache=True, attention_mask=attention_mask)
        kv_all = self._steal_kv_cache()

        # Log-probs accumulator
        token_lps = [] if return_logprobs else None

        # Sample first token
        raw_logits = logits_full[:, -1, :]  # pre-temperature
        next_logits = raw_logits / temperature
        if top_k > 0:
            v, _ = next_logits.topk(top_k)
            next_logits[next_logits < v[:, -1:]] = float("-inf")
        probs = torch.softmax(next_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        if return_logprobs:
            token_lps.append(F.log_softmax(raw_logits, dim=-1).gather(-1, next_tok))
        all_ids = torch.cat([input_ids, next_tok], dim=1)

        # Incremental generation
        gen_count = 1
        for _ in range(1, max_new_tokens):
            new_tok = all_ids[:, -1:]  # (B, 1)
            pos_value = prompt_len + gen_count - 1
            pos_ids = torch.full((B, 1), pos_value, dtype=torch.long, device=device)

            h = self.embed_tokens(new_tok)
            pos_embeds = self._compute_position_embeddings(h, pos_ids)

            # Per-Layer Embeddings for the new token (E2B/E4B models)
            ple_new = None
            if hasattr(self, "_ple_enabled") and self._ple_enabled:
                ple_new = self._compute_ple(new_tok, h)

            # Fresh shared KV per token (layers overwrite, not append)
            shared_kv = collections.UserDict()

            # Prelude: incremental with cached KV
            h = self._run_block_incremental(
                h, self.prelude_indices, pos_embeds, None, pos_ids,
                kv_cache=kv_all["prelude"], per_layer_inputs=ple_new,
                shared_kv_states=shared_kv)

            e = h.detach()

            # Recurrent: loop via shared _recurrent_loop method
            def _run_rec_inc(h, t):
                return self._run_block_incremental(
                    h, self.recurrent_indices, pos_embeds, None, pos_ids,
                    kv_cache=kv_all["recurrent"][t], per_layer_inputs=ple_new,
                    shared_kv_states=shared_kv,
                )

            h, _, _ = self._recurrent_loop(h, e, n_loops, _run_rec_inc, return_kv_cache=False)

            # Coda: incremental with cached KV
            h = self._run_block_incremental(
                h, self.coda_indices, pos_embeds, None, pos_ids,
                kv_cache=kv_all["coda"], per_layer_inputs=ple_new,
                shared_kv_states=shared_kv)

            h = self.norm(h)
            logits = self.lm_head(h)  # (B, 1, V)
            if self._logit_softcap is not None:
                logits = self._logit_softcap * torch.tanh(logits.float() / self._logit_softcap).to(logits.dtype)

            # Sample
            raw_logits = logits[:, -1, :]  # pre-temperature
            next_logits = raw_logits / temperature
            if top_k > 0:
                v, _ = next_logits.topk(top_k)
                next_logits[next_logits < v[:, -1:]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            if return_logprobs:
                token_lps.append(F.log_softmax(raw_logits, dim=-1).gather(-1, next_tok))
            all_ids = torch.cat([all_ids, next_tok], dim=1)
            gen_count += 1

            if (next_tok == self._eos_id).all():
                break

        if return_logprobs:
            return all_ids, torch.cat(token_lps, dim=-1)  # (B, L_new, 1) -> (B, L_new)
        return all_ids

    def _steal_kv_cache(self) -> dict:
        """Return KV caches captured during the last forward() call.

        prelude and coda caches from _last_present_kv on attention modules.
        recurrent caches from self._kv_rec (list per iteration).
        """
        def _get_kv(idx):
            layer = self._layer_module_list[idx]
            attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
            return getattr(attn, '_last_present_kv', None) if attn else None

        prelude_kv = {idx: _get_kv(idx) for idx in self.prelude_indices}
        coda_kv = {idx: _get_kv(idx) for idx in self.coda_indices}
        rec_kv = getattr(self, '_kv_rec', [None] * self._last_n_loops)
        return {"prelude": prelude_kv, "coda": coda_kv, "recurrent": rec_kv}

    def _run_block_incremental(
        self,
        h: torch.Tensor,
        layer_indices: list[int],
        position_embeddings: dict,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        kv_cache: dict,
        per_layer_inputs: Optional[torch.Tensor] = None,
        shared_kv_states: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run block on a single token with KV cache.

        kv_cache maps layer_idx -> past_key_value tuple (or None).
        Updated past_key_values are written back in-place.
        """
        for idx in layer_indices:
            layer = self._layer_module_list[idx]
            layer_type = self._hf_config.text_config.layer_types[idx]
            ple = None
            if per_layer_inputs is not None:
                ple = per_layer_inputs[:, :, idx, :]
            pkv = kv_cache.get(idx)
            layer_out = layer(
                hidden_states=h,
                per_layer_input=ple,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=pkv,
                use_cache=True,
            )
            # h may be tuple (hidden_states, present_key_value) if use_cache=True
            if isinstance(layer_out, tuple):
                h, pkv_new = layer_out
                kv_cache[idx] = pkv_new
            else:
                h = layer_out
        return h

    def trainable_parameters(self):
        """Return only the new (non-pretrained) parameters for training."""
        params = []
        params.extend(self.injection.parameters())
        if self.depth_lora is not None:
            params.extend(self.depth_lora.parameters())
        return params

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        pretrained = total - trainable
        return {
            "total": total,
            "pretrained": pretrained,
            "trainable": trainable,
        }
