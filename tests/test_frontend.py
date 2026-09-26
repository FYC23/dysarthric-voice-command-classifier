"""Tests for BC-ResNet's log-Mel input features and SpecAugment."""

import pytest
import torch

from src.model.bcresnet import BCResNet
from src.model.frontend import LogMel, SpecAugParams, spec_augment, spec_augment_params

SR = 16000


class TestLogMel:
    @pytest.mark.parametrize("seconds, frames", [(1, 101), (2, 201)])
    def test_shape(self, seconds, frames):
        assert LogMel()(torch.randn(3, seconds * SR) * 0.1).shape == (3, 1, 40, frames)

    def test_silence_is_finite(self):
        assert torch.isfinite(LogMel()(torch.zeros(1, SR))).all()

    def test_rejects_unbatched_input(self):
        with pytest.raises(ValueError, match="batch"):
            LogMel()(torch.zeros(SR))

    def test_feeds_bc_resnet(self):
        logits = BCResNet(tau=1, num_classes=12)(LogMel()(torch.randn(2, 2 * SR) * 0.1))
        assert logits.shape == (2, 12)


class TestSpecAugmentParams:
    def test_follows_the_paper_by_width(self):
        assert spec_augment_params(1) is None
        assert [spec_augment_params(t).freq_mask_param for t in (1.5, 2, 3, 6, 8)] == [1, 3, 5, 7, 7]
        assert spec_augment_params(8).time_mask_param == 20

    def test_unknown_width_raises(self):
        with pytest.raises(ValueError, match="tau"):
            spec_augment_params(4)

    def test_time_masks_keep_the_papers_density_of_two_per_second(self):
        assert spec_augment_params(8).num_time_masks == 2
        assert spec_augment_params(8, window_s=2.0).num_time_masks == 4
        assert spec_augment_params(2, window_s=2.0).num_freq_masks == 2

    def test_width_one_still_has_no_specaugment_at_2s(self):
        assert spec_augment_params(1, window_s=2.0) is None

    @pytest.mark.parametrize("window_s", [0, -1.0])
    def test_rejects_non_positive_window(self, window_s):
        with pytest.raises(ValueError, match="window"):
            spec_augment_params(8, window_s=window_s)


class TestSpecAugment:
    PARAMS = SpecAugParams(freq_mask_param=7)

    def masked(self, seed, batch=8):
        features = torch.ones(batch, 1, 40, 201)
        return features, spec_augment(features, self.PARAMS, torch.Generator().manual_seed(seed))

    def test_masks_are_bounded(self):
        _, out = self.masked(0)
        for example in out[:, 0]:
            zero_rows = (example == 0).all(dim=1).sum().item()
            zero_cols = (example == 0).all(dim=0).sum().item()
            assert zero_rows <= 2 * 6   # 2 masks, width < 7
            assert zero_cols <= 2 * 19  # 2 masks, width < 20

    def test_does_not_mutate_input(self):
        features, _ = self.masked(0)
        assert (features == 1).all()

    def test_each_example_gets_its_own_masks(self):
        _, out = self.masked(0)
        assert any(not torch.equal(out[0], out[i]) for i in range(1, 8))

    def test_reproducible_from_the_generator(self):
        assert torch.equal(self.masked(3)[1], self.masked(3)[1])

    def test_zero_frequency_param_masks_no_frequencies(self):
        params = SpecAugParams(freq_mask_param=0)
        out = spec_augment(torch.ones(4, 1, 40, 201), params, torch.Generator().manual_seed(0))
        assert (out[:, 0] == 0).all(dim=2).sum() == 0

    def test_four_time_masks_stay_bounded(self):
        params = SpecAugParams(freq_mask_param=7, num_time_masks=4)
        out = spec_augment(torch.ones(8, 1, 40, 201), params, torch.Generator().manual_seed(0))
        for example in out[:, 0]:
            assert (example == 0).all(dim=0).sum().item() <= 4 * 19
