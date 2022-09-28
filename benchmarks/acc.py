import gym
import numpy as np
from typing import Tuple, Dict, Any


class AccEnv(gym.Env):

    def __init__(self):
        super(AccEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(1,))
        self.observation_space = gym.spaces.Box(low=np.array([-10, -10]),
                                                high=np.array([10, 10]))

        self.init_space = gym.spaces.Box(low=np.array([-1.1, -0.1]),
                                         high=np.array([-0.9, 0.1]))
        self.state = np.zeros(2)

        self.rng = np.random.default_rng()

        self.concrete_safety = [
            lambda x: x[0]
        ]

        self._max_episode_steps = 300

        self.polys = [
            # P (s 1)^T <= 0 iff s[0] >= 0
            # P = [[-1, 0, 0]]
            np.array([[-1.0, 0.0, 0.0]])
        ]

        self.safe_polys = [
            np.array([[1.0, 0.0, 0.01]])
        ]

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.02
        x = self.state[0] + dt * self.state[1]
        v = self.state[1] + dt * \
            (action[0] + self.rng.normal(loc=0, scale=0.5))
        self.state = np.clip(
                np.asarray([x, v]),
                self.observation_space.low,
                self.observation_space.high)
        reward = 2.0 + x if x < 0 else -10
        done = bool(x >= 0) or self.steps > self._max_episode_steps
        self.steps += 1
        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return state[0] >= 0

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)
        self.rng = np.random.default_rng(np.random.PCG64(seed))

    def unsafe(self, state: np.ndarray) -> bool:
        return state[0] >= 0
