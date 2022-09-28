import gym
import numpy as np
from typing import Tuple, Dict, Any


class RoadEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-20, high=20, shape=(2,))

        self.init_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,))

        self.max_speed = 10.0

        self._max_episode_steps = 300

        self.polys = [
            np.array([[0.0, 1.0, self.max_speed]]),
            np.array([[0.0, -1.0, -self.max_speed]])
        ]

        self.safe_polys = [
            np.array([[0.0, 1.0, -self.max_speed + 0.01],
                      [0.0, -1.0, self.max_speed + 0.01]])
        ]

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.02
        x = self.state[0] + dt * self.state[1]
        vx = self.state[1] + dt * action[0]

        self.state = np.array([x, vx])
        self.state = np.clip(self.state, self.observation_space.low,
                             self.observation_space.high)

        reward = -abs(x - 3.0)
        done = x >= 3.0 or self.steps >= self._max_episode_steps or \
            self.unsafe(self.state)
        self.steps += 1

        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return state[0] >= 3.0

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self, state: np.ndarray) -> bool:
        return state[1]**2 >= self.max_speed**2
