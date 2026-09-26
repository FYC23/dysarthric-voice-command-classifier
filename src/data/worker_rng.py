"""One numpy generator per DataLoader process, for on-the-fly augmentation."""

from typing import Optional

import numpy as np
import torch


class WorkerRng:
    """
    A numpy generator seeded from torch. DataLoader workers each have their own
    torch seed, so a generator copied in from the main process is replaced, not
    reused. With num_workers=0, set_seed() makes the draws reproducible.
    """

    def __init__(self):
        self._rng: Optional[np.random.Generator] = None
        self._seed: Optional[int] = None

    def get(self) -> np.random.Generator:
        seed = torch.initial_seed()
        if self._rng is None or self._seed != seed:
            self._rng = np.random.default_rng(seed % 2**32)
            self._seed = seed
        return self._rng
