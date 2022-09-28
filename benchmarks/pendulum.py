import gym
import numpy as np
from typing import Tuple, Dict, Any


class PendulumEnv(gym.Env):

    def __init__(self):
        super(PendulumEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(
                low=np.array([-np.pi, -1]),
                high=np.array([np.pi, 1]))

        self.init_space = gym.spaces.Box(low=np.array([-0.01, -0.001]),
                                         high=np.array([0.01, 0.001]))

        self.state = np.zeros(2)

        self.unsafe_reward = -10

        self._max_episode_steps = 100

        self.polys = [
            np.array([[1.0, 0.0, 0.4]]),
            np.array([[-1.0, 0.0, 0.4]])
        ]

        self.safe_polys = [
            np.array([[-1.0, 0.0, -0.39],
                      [1.0, 0.0, -0.39]])
        ]

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.02
        m = 0.25
        ln = 2.0
        g = 9.81
        theta = self.state[0] + dt * self.state[1]
        omega = self.state[1] + \
            dt * (g / (2 * ln) * np.sin(self.state[0]) +
                  3.0 / (m * ln ** 2) * action[0])
        self.state = np.clip(
                np.asarray([theta, omega]),
                self.observation_space.low,
                self.observation_space.high)
        reward = -abs(theta)
        done = bool(abs(theta) >= 0.4) or \
            self.steps >= self._max_episode_steps or self.unsafe(self.state)
        self.steps += 1
        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return abs(state[0]) >= 0.4

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self, state: np.ndarray) -> bool:
        return abs(self.state[0]) >= 0.4
