import torch

from scripts.train_srpo import grpo_loss


def test_grpo_mixed_group_has_gradient():
    log_probs = torch.zeros(2, 3, requires_grad=True)
    log_probs_old = torch.zeros(2, 3)
    rewards = torch.tensor([1.0, 0.0])
    response_mask = torch.ones(2, 3)
    group_ids = torch.tensor([0, 0])

    loss = grpo_loss(
        log_probs,
        log_probs_old,
        rewards,
        response_mask,
        epsilon=0.2,
        epsilon_high=0.28,
        group_ids=group_ids,
    )
    loss.backward()

    assert log_probs.grad is not None
    assert log_probs.grad.abs().sum() > 0


def test_grpo_uniform_group_has_no_policy_signal():
    log_probs = torch.zeros(2, 3, requires_grad=True)
    log_probs_old = torch.zeros(2, 3)
    rewards = torch.tensor([1.0, 1.0])
    response_mask = torch.ones(2, 3)
    group_ids = torch.tensor([0, 0])

    loss = grpo_loss(
        log_probs,
        log_probs_old,
        rewards,
        response_mask,
        epsilon=0.2,
        epsilon_high=0.28,
        group_ids=group_ids,
    )
    loss.backward()

    assert log_probs.grad is not None
    assert torch.equal(log_probs.grad, torch.zeros_like(log_probs.grad))
