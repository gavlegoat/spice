import gym
import numpy as np
from typing import Tuple, Dict, Any


class NoisyRoad2dEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-20, high=20, shape=(4,))

        self.init_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(4,))

        self.rng = np.random.default_rng()

        self.max_speed = 10.0

        n = 8
        xs = []
        ys = []
        for i in range(n):
            ang = 2 * i * np.pi / n + np.pi / n
            xs.append(self.max_speed * np.cos(ang))
            ys.append(self.max_speed * np.sin(ang))

        self._max_episode_steps = 300

        self.polys = []
        for i in range(n):
            j = (i + 1) % n
            if abs(xs[j] - xs[i]) < 1e-6:
                if xs[i] >= 0.0:
                    self.polys.append(np.array([[0.0, 0.0, -1.0, 0.0, xs[i]]]))
                else:
                    self.polys.append(np.array([[0.0, 0.0, 1.0, 0.0, -xs[i]]]))
            else:
                m = (ys[j] - ys[i]) / (xs[j] - xs[i])
                b = ys[i] - m * xs[i]
                if b >= 0.0:
                    self.polys.append(np.array([[0.0, 0.0, m, -1.0, b]]))
                else:
                    self.polys.append(np.array([[0.0, 0.0, -m, 1.0, -b]]))

        mat = np.zeros((n, 5))
        for i in range(n):
            j = (i + 1) % n
            if abs(xs[j] - xs[i]) < 1e-6:
                if xs[i] >= 0.0:
                    mat[i, 2] = 1.0
                    mat[i, 4] = -xs[i]
                else:
                    mat[i, 2] = -1.0
                    mat[i, 4] = xs[i]
            else:
                m = (ys[j] - ys[i]) / (xs[j] - xs[i])
                b = ys[i] - m * xs[i]
                if b >= 0.0:
                    mat[i, 2] = -m
                    mat[i, 3] = 1.0
                    mat[i, 4] = -b + 0.01
                else:
                    mat[i, 2] = m
                    mat[i, 3] = -1.0
                    mat[i, 4] = b + 0.01
        self.safe_polys = [mat]

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

        self.state = np.array([x, y, vx, vy])
        self.state += self.rng.normal(loc=0, scale=0.05, size=(4,))
        self.state = np.clip(self.state, self.observation_space.low,
                             self.observation_space.high)

        reward = -(abs(x - 3.0) + abs(y - 3.0))
        done = x >= 3.0 and y >= 3.0
        done = done or self.steps >= self._max_episode_steps or \
            self.unsafe(self.state)
        self.steps += 1

        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return state[0] >= 3.0 and state[1] >= 3.0

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)
        self.rng = np.random.default_rng(np.random.PCG64(seed))

    def unsafe(self, state: np.ndarray) -> bool:
        return state[0]**2 + state[1]**2 >= self.max_speed**2
