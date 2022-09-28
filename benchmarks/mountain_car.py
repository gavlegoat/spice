import gym
import numpy as np
from typing import Tuple, Dict, Any


class MountainCarEnv(gym.Env):

    def __init__(self):
        super(MountainCarEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(
                low=np.array([-1.2, -7]),
                high=np.array([0.7, 7]))

        self.init_space = gym.spaces.Box(
                low=np.array([-0.55, -0.1]),
                high=np.array([-0.45, 0.1]))

        self.state = np.zeros(2)

        self.safe_limit = -np.pi / 3 - 0.1

        self.unsafe_reward = -100

        self._max_episode_steps = 300

        self.polys = [
            np.array([[1.0, 0.0, self.safe_limit]])
        ]

        self.safe_polys = [
            np.array([[-1.0, 0.0, -self.safe_limit]])
        ]

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        self.best_x = self.state[0]
        return self.state

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.05
        x = self.state[0] + dt * self.state[1]
        v = self.state[1] + dt * (action[0] - 2.5 * np.cos(3 * self.state[0]))
        self.state = np.clip(
                np.array([x, v]),
                self.observation_space.low,
                self.observation_space.high)
        reward = -1.0
        done = bool(x >= 0.6) or self.steps >= self._max_episode_steps
        self.steps += 1
        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return state[0] >= 0.6 or state[0] < -np.pi / 3

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self, state: np.ndarray) -> bool:
        return self.state[0] < self.safe_limit
