"""Windows/single-GPU entry point for SRPO training.

This intentionally reuses ``train_srpo.py`` instead of carrying a second copy
of the training loop.  The overrides below are only hardware sizing choices;
the model, reward, GRPO, and SDPO logic stay identical to the main trainer.
"""

from train_srpo import SRPOTrainer, TrainConfig


def main():
    cfg = TrainConfig()
    cfg.max_loops = 4
    cfg.micro_batch_size = 1
    cfg.gradient_accumulation_steps = 8
    cfg.total_steps = 500
    cfg.save_every = 200
    cfg.eval_every = 50
    cfg.log_every = 10
    trainer = SRPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
