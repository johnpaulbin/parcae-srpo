from types import SimpleNamespace

import torch

from scripts.train_srpo import SRPOTrainer, TrainConfig
from parcae.model import _find_first_module_path, _resolve_module_path


def test_resolve_nested_module_path():
    leaf = object()
    root = SimpleNamespace(base_model=SimpleNamespace(model=SimpleNamespace(language_model=leaf)))

    assert _resolve_module_path(root, "base_model.model.language_model") is leaf
    assert _resolve_module_path(root, "base_model.missing.language_model") is None


def test_find_first_module_path_skips_missing_wrappers():
    leaf = object()
    root = SimpleNamespace(base_model=SimpleNamespace(model=SimpleNamespace(language_model=leaf)))

    path, value = _find_first_module_path(
        root,
        ["model.language_model", "base_model.model.language_model"],
    )

    assert path == "base_model.model.language_model"
    assert value is leaf


def test_training_view_keeps_full_sequence_when_disabled():
    trainer = object.__new__(SRPOTrainer)
    trainer.cfg = TrainConfig(max_train_sequence_tokens=0)

    ids = torch.arange(10)
    view, prompt_len = trainer._training_view(ids, prompt_len=4)

    assert torch.equal(view, ids)
    assert prompt_len == 4


def test_training_view_crops_left_and_rebases_prompt_len():
    trainer = object.__new__(SRPOTrainer)
    trainer.cfg = TrainConfig(max_train_sequence_tokens=6)

    ids = torch.arange(10)
    view, prompt_len = trainer._training_view(ids, prompt_len=8)

    assert torch.equal(view, torch.arange(4, 10))
    assert prompt_len == 4


def test_training_view_inside_completion_masks_prompt_to_zero():
    trainer = object.__new__(SRPOTrainer)
    trainer.cfg = TrainConfig(max_train_sequence_tokens=6)

    ids = torch.arange(10)
    view, prompt_len = trainer._training_view(ids, prompt_len=3)

    assert torch.equal(view, torch.arange(4, 10))
    assert prompt_len == 0
