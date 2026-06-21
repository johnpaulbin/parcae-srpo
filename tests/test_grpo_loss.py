import torch

from scripts.train_srpo import grpo_loss, sdpo_loss


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


def test_sdpo_loss_ignores_prompt_tokens():
    response_mask = torch.tensor([[0.0, 0.0, 1.0]])
    student = torch.zeros(1, 3, 4, requires_grad=True)
    teacher = torch.zeros(1, 3, 4)
    teacher[:, 2, 1] = 5.0

    baseline = sdpo_loss(student, teacher, response_mask, entropy_weight=0.01)

    student_with_bad_prompt = student.detach().clone().requires_grad_(True)
    teacher_with_bad_prompt = teacher.clone()
    with torch.no_grad():
        student_with_bad_prompt[:, :2, 0] = 50.0
    teacher_with_bad_prompt[:, :2, 3] = 50.0

    changed_prompt = sdpo_loss(
        student_with_bad_prompt,
        teacher_with_bad_prompt,
        response_mask,
        entropy_weight=0.01,
    )

    assert torch.allclose(baseline, changed_prompt)


def test_sdpo_loss_with_no_response_tokens_is_zero_with_grad():
    student = torch.zeros(1, 2, 4, requires_grad=True)
    teacher = torch.zeros(1, 2, 4)
    response_mask = torch.zeros(1, 2)

    loss = sdpo_loss(student, teacher, response_mask, entropy_weight=0.01)
    loss.backward()

    assert loss.item() == 0.0
    assert student.grad is not None
    assert torch.equal(student.grad, torch.zeros_like(student.grad))
