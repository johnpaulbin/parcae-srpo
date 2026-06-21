import json

from scripts.train_srpo import SRPOTrainer, TrainConfig


def test_sample_logging_preserves_full_prompt_and_completion(tmp_path, capsys):
    trainer = object.__new__(SRPOTrainer)
    cfg = TrainConfig()
    cfg.sample_log_prompts = 1
    cfg.sample_log_path = str(tmp_path / "samples.jsonl")
    trainer.cfg = cfg

    prompt = "def solve(x):\n    \"\"\"Return x unchanged.\"\"\"\n"
    model_prompt = "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n"
    completion = "def solve(x):\n    return x\n# no truncation marker at the end"
    feedback = "passed all tests"

    samples = trainer._collect_samples(
        [prompt, "def other():\n    pass\n"],
        [model_prompt, "template for other"],
        [
            {
                "batch_idx": 0,
                "text": completion,
                "reward": 1,
                "feedback": feedback,
            },
            {
                "batch_idx": 1,
                "text": "def other():\n    return None\n",
                "reward": 0,
                "feedback": "not selected",
            },
        ],
    )

    trainer._log_samples(
        7,
        {
            "samples": samples,
            "T": 3,
            "loss": 1.25,
            "grpo_loss": 0.5,
            "sdpo_loss": 0.75,
            "reward_mean": 1.0,
            "n_correct": 1,
            "n_failed": 0,
            "rho": 0.9,
        },
    )

    output = capsys.readouterr().out
    assert prompt in output
    assert model_prompt in output
    assert completion in output
    assert feedback in output
    assert "grpo=0.5000" in output
    assert "sdpo=0.7500" in output

    rows = [
        json.loads(line)
        for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["prompt"] == prompt
    assert rows[0]["model_prompt"] == model_prompt
    assert rows[0]["completion"] == completion
    assert rows[0]["feedback"] == feedback
    assert rows[0]["grpo_loss"] == 0.5
    assert rows[0]["sdpo_loss"] == 0.75
