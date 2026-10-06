"""
Feature extractors (student assignment placeholders).

This file contains lightweight, instructor-friendly skeletons for feature
extractors. Students should implement the missing functionality as part of
the assignment. The goal is to provide clear method signatures, small
helper methods, and inline hints, but not a full implementation.

HINTS:
- Keep the public API (class names and method names) stable so tests and
  other modules can import these types.
- Implementations should work for the MountainCar 2D state: [position, velocity].
  Use `env.observation_space.low` and `.high` to obtain bounds.
"""

from abc import ABC, abstractmethod
from math import ceil, sqrt
import numpy as np
import gymnasium as gym


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors.

    Students should implement `num_features` and `extract_features`.
    A minimal `normalize_state` helper is provided and may be used.
    """

    def __init__(self, env: gym.Env):
        self.env = env
        # Derived classes should set/compute their own attributes
        # (for example, centers for RBF).
        self.n_features = self.num_features()

        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.state_bounds = list(zip(self.obs_low, self.obs_high))
        self.position_bounds = (self.obs_low[0], self.obs_high[0])
        self.velocity_bounds = (self.obs_low[1], self.obs_high[1])

    @abstractmethod
    def num_features(self) -> int:
        """Return the number of features in the feature vector."""
        pass

    @abstractmethod
    def extract_features(self, state: np.ndarray) -> np.ndarray:
        """Return feature vector for a given state.

        Args:
            state: raw environment state (e.g., [position, velocity])
        """
        pass

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize a state into [0, 1] for each dimension.

        This helper clamps out-of-bounds values to the [0,1] interval which
        is useful for grid-based or RBF centers defined in normalized space.
        """
        max_value = self.obs_high
        min_value = self.obs_low
        state = (state-min_value)/(max_value-min_value)
        state = np.clip(state, 0, 1)
        return state



class RBFFeatureExtractor(FeatureExtractor):
    """Skeleton RBF feature extractor for students to implement.

    Required tasks for students:
    - Implement `_create_rbf_centers` to place `n_centers` centers in normalized
      state space (0..1). A grid layout is a good starting point.
    - Implement `num_features` to return the number of centers.
    - Implement `extract_features` to compute Gaussian RBFs: exp(-||x-c||^2/(2*sigma^2)).

    HINTS:
    - Use `self.normalize_state(state)` to map raw states into [0,1].
    - If `n_centers` is not a perfect square, create a grid with at least
      ceil(sqrt(n_centers)) per dimension and then slice to the desired count.
    - Ensure the returned feature vector has length `self.n_features`.
    """

    def __init__(self, env: gym.Env, n_centers: int = 25, sigma: float = 0.1):
        # Student-implemented: store parameters and create centers
        self.n_centers = int(n_centers)
        self.sigma = float(sigma)

        
        self._create_rbf_centers()

        super().__init__(env)

    def num_features(self) -> int:
        # Students should return the number of RBF centers here.
        return int(self.n_centers)

    def _create_rbf_centers(self) -> None:
        """(Student) Create RBF centers in normalized state space.

        Students: implement this method to populate `self.centers`.
        The centers should be numpy array of shape (n_centers, 2) evenly distributed
        within the normalized [0,1] coordinates. The first center must be at (0,0) and the last at (1,1).

        """
        dimension_center_numbers = ceil(sqrt(self.n_centers))
        dimension_points = []
        for center in range(dimension_center_numbers):
            dimension_points.append(center/(dimension_center_numbers-1))

        dimension_centers = []
        for i in dimension_points:
            for j in dimension_points:
                dimension_centers.append((i,j))

        dimension_centers = np.array(dimension_centers)
        dimension_centers = dimension_centers[:self.n_centers]
        self.centers = dimension_centers


    def extract_features(self, state: np.ndarray) -> np.ndarray:
        """(Student) Compute RBF feature activations for a state.

        Returns a 1D numpy array of length `self.n_features`.
        """
        normalized_state = self.normalize_state(state)
        centers = self.centers
        diff = normalized_state - centers
        dist = np.sum(diff**2,axis=1)
        return np.exp(-dist/(2*self.sigma**2))
