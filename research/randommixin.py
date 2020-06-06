"""Encapsulate a randomized object."""

from random import Random, random
from typing import Any


class RandomMixin:
    """Encapsulation of a randomized object."""

    def __init__(self, random_seed=None):
        # type: (Any) -> None
        """Initialize the RandomClass.

        Arguments:
            random_seed (Any): The random seed.
        """
        if random_seed is None:
            random_seed = random()
        self.random_seed = random_seed
        self.rng = Random(self.random_seed)
        super().__init__()
