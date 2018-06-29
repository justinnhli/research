"""Encapsulate a randomized object."""

from random import Random, random


class RandomMixin:
    """Encapsulation of a randomized object."""

    def __init__(self, random_seed=None):
        """Initialize the RandomClass.

        Arguments:
            random_seed (Object): The random seed.
        """
        if random_seed is None:
            random_seed = random()
        self.random_seed = random_seed
        self.rng = Random(self.random_seed)
