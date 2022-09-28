import gym
import numpy as np
from typing import Tuple, Dict, Any


class MidObstacleEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(4,))

        self.init_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(4,))

        self._max_episode_steps = 400

        self.polys = [
            np.array([[1.0, 0.0, 0.0, 0.0, -2.0],
                      [-1.0, 0.0, 0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0, 0.0, -2.0],
                      [0.0, -1.0, 0.0, 0.0, 1.0]])
        ]

        self.safe_polys = [
            np.array([[1.0, 0.0, 0.0, 0.0, -0.99]]),
            np.array([[-1.0, 0.0, 0.0, 0.0, 2.01]]),
            np.array([[0.0, 1.0, 0.0, 0.0, -0.99]]),
            np.array([[0.0, -1.0, 0.0, 0.0, 2.01]])
        ]

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.02
        x = self.state[0] + dt * self.state[2]
        y = self.state[1] + dt * self.state[3]
        vx = self.state[2] + dt * action[0]
        vy = self.state[3] + dt * action[1]

        self.state = np.clip(np.array([x, y, vx, vy]),
                             self.observation_space.low,
                             self.observation_space.high)

        reward = -(abs(x - 3.0) + abs(y - 3.0))

        done = x >= 3.0 and y >= 3.0
        done = done or self.steps > self._max_episode_steps
        self.steps += 1

        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return state[0] >= 3.0 and state[1] >= 3.0

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self, state: np.ndarray) -> bool:
        return self.state[0] >= 1.0 and self.state[0] <= 2.0 and \
            self.state[1] >= 1.0 and self.state[1] <= 2.0
