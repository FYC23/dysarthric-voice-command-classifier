"""The paper's learning-rate schedule: linear warmup from 0, then cosine to 0, per step."""

import pytest
import torch

from src.training.schedule import warmup_cosine, warmup_cosine_scheduler


def test_rises_linearly_then_decays_to_zero():
    f = warmup_cosine(total_steps=100, warmup_steps=10)
    assert f(0) == 0.0
    assert f(5) == pytest.approx(0.5)
    assert f(10) == pytest.approx(1.0)
    assert f(55) == pytest.approx(0.5)
    assert f(100) == pytest.approx(0.0)
    assert all(f(s) >= f(s + 1) for s in range(10, 100))


def test_no_warmup_starts_at_the_peak():
    assert warmup_cosine(total_steps=10, warmup_steps=0)(0) == 1.0


def test_steps_past_the_end_stay_at_zero():
    assert warmup_cosine(total_steps=10, warmup_steps=2)(15) == 0.0


@pytest.mark.parametrize("total, warmup", [(0, 0), (10, 11), (10, -1)])
def test_invalid_lengths_raise(total, warmup):
    with pytest.raises(ValueError):
        warmup_cosine(total, warmup)


def test_scheduler_scales_the_optimizer_learning_rate():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    sched = warmup_cosine_scheduler(opt, total_steps=4, warmup_steps=2)
    lrs = []
    for _ in range(4):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs == pytest.approx([0.0, 0.05, 0.1, 0.05])
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0)
