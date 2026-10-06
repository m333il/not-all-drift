import torch

from prompt_optimization.early_exit import ResidualLinear, forward_kl


def test_residual_linear_starts_at_mean_shift():
    mean = torch.tensor([1.0, -2.0])
    model = ResidualLinear(mean)
    values = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(model.state_prediction(values), values + mean)


def test_forward_kl_is_zero_for_equal_logits():
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    assert torch.allclose(forward_kl(logits, logits), torch.zeros(1), atol=1e-7)
