import gym
import numpy as np
from typing import Optional, List, Tuple
import cvxopt
import scipy
import torch

from .env_model import MARSModel
from pytorch_soft_actor_critic.sac import SAC
from pytorch_soft_actor_critic.replay_memory import ReplayMemory


cvxopt.solvers.options['show_progress'] = False


class SACPolicy:

    def __init__(self,
                 gym_env: gym.Env,
                 replay_size: int,
                 seed: int,
                 batch_size: int,
                 sac_args):
        self.agent = SAC(gym_env.observation_space.shape[0],
                         gym_env.action_space, sac_args)
        self.memory = ReplayMemory(replay_size, seed)
        self.updates = 0
        self.batch_size = batch_size

    def __call__(self, state: np.ndarray, evaluate: bool = False):
        return self.agent.select_action(state, evaluate=evaluate)

    def add(self, state, action, reward, next_state, done, cost):
        self.memory.push(state, action, reward, next_state, done, cost)

    def train(self):
        ret = self.agent.update_parameters(self.memory, self.batch_size,
                                           self.updates)
        self.updates += 1
        return ret

    def report(self):
        return 0, 0

    def load_checkpoint(self, path):
        self.agent.load_checkpoint(path)


class ProjectionPolicy:
    """
    Wrap an underlying policy in a safety layer based on prejection onto a
    weakest precondition.
    """

    def __init__(self,
                 env: MARSModel,
                 state_space: gym.Space,
                 action_space: gym.Space,
                 horizon: int,
                 unsafe_polys: List[np.ndarray],
                 safe_polys: List[np.ndarray]):
        self.env = env
        self.horizon = horizon
        self.state_space = state_space
        self.action_space = action_space
        self.unsafe_polys = unsafe_polys
        self.safe_polys = safe_polys
        self.saved_state = None
        self.saved_action = None

    def backup(self, state: np.ndarray) -> np.ndarray:
        """
        Choose a backup action if the projection fails.
        """
        s_dim = self.state_space.shape[0]
        P = cvxopt.spmatrix(1.0, range(s_dim), range(s_dim))
        q = cvxopt.matrix(0.0, (s_dim, 1))

        best_val = 1e10
        best_proj = np.zeros_like(state)
        for unsafe_mat in self.unsafe_polys:
            tmp = unsafe_mat[:, :-1]
            G = cvxopt.matrix(tmp)
            h = cvxopt.matrix(-unsafe_mat[:, -1] - np.dot(tmp, state))
            sol = cvxopt.solvers.qp(P, q, G, h)
            soln = np.asarray(sol['x']).squeeze()
            if len(soln.shape) == 0:
                soln = soln[None]
            val = np.linalg.norm(soln)
            if val < best_val:
                best_val = val
                best_proj = soln
        best_proj = best_proj / np.linalg.norm(best_proj)
        u_dim = self.action_space.shape[0]
        point = np.concatenate((state, np.zeros(u_dim)))
        mat, _ = self.env.get_matrix_at_point(point, s_dim)
        # M is the concatenated linear model, so we need to split it into the
        # dynamics and the input
        A = mat[:, :s_dim]
        B = mat[:, s_dim:-1]
        # c = mat[:, -1]   # c only contributes a constant and isn't needed
        # Some analysis:
        # x_{i+1} = A x_i + B u_i + c
        # x_1 = A x_0 + B u_0 + c
        # x_2 = A (A x_0 + B u_0 + c) + B u_1 + c
        #     = A^2 x_0 + A B u_0 + A c + B u_1 + c
        # x_3 = A (A^2 x_0 + A B u_0 + A c + B u_1 + c) + B u_2 + c
        #     = A^3 x_0 + A^2 B u_0 + A^2 c + A B u_1 + A c + B u_2 + c
        # x_i = A^i x_0 + A^{i-1} B u_0 + ... + A B u_{i-2} + B u_{i-1} +
        #           A^{i-1} c + ... + A c + c
        # x_H = A^H x_0
        #     + \sum_{i=0}^{H-1} A^{H-i-1} B u_i
        #     + \sum_{i=1}^{H-1} A^i c
        # Now we maximize -best_proj^T (x_H - x_0). -best_proj^T x_0 is
        # constant so we can igonore it and just maximize -best_proj^T x_H.
        # (let q = -best_proj for convenience)
        #   q^T x_H
        # = q^T (A^H x_0 + sum A^i c + sum A^{H-i-1} B u_i)
        # = q^T A^H x_0 + q^T sum A^i c + q^T sum A^{H-i-1} B u_i
        # We can remove the constants q^T A^H x_0 and q^T sum A^i c
        # maximize q^T sum A^{H-i-1} B u_i
        #    = sum q^T A^{H-i-1} B u_i
        #    = sum (q^T A^{H-i-1} B) u_i
        #    = sum ((A^{H-i-1} B)^T q)^T u_i
        # So in the end, let
        # m = [((A^{H-1} B)^T q)^T
        #      ((A^{H-2} B)^T q)^T
        #      ...
        #      ((A B)^T q)^T
        #      (B^T q)^T]
        # and let u = [u_0 u_1 ... u_{H-1}]. Then we need to solve
        # maximize m^T u subject to action space constraints
        m = np.zeros(self.horizon * u_dim)
        for i in range(self.horizon):
            m[u_dim*i:u_dim*(i+1)] = \
                np.dot(np.dot(np.linalg.matrix_power(A, self.horizon - i - 1),
                              B).T, -best_proj).T
        act_bounds = np.stack((self.action_space.low, self.action_space.high),
                              axis=1)
        bounds = np.concatenate([act_bounds] * self.horizon, axis=0)
        # linprog minimizes, so we need -m here
        res = scipy.optimize.linprog(-m, bounds=bounds)
        # Return the first action
        return res['x'][:u_dim]

    def solve(self,    # noqa: C901
              state: np.ndarray,
              action: Optional[np.ndarray] = None,
              debug: bool = False) -> np.ndarray:
        """
        Solve the synthesis problem and store the result. This sets the saved
        action and state because very often we will call unsafe and then
        __call__ on the same state.
        """
        s_dim = self.state_space.shape[0]
        u_dim = self.action_space.shape[0]
        # If we don't have a proposed action, look for actions with small
        # magnitude
        if action is None:
            action = np.zeros(u_dim)
        # Get the local dynamics
        point = np.concatenate((state, action))
        mat, eps = self.env.get_matrix_at_point(point, s_dim)
        A = mat[:, :s_dim]
        B = mat[:, s_dim:-1]
        c = mat[:, -1]

        best_score = 1e10
        best_u0 = None

        for poly in self.safe_polys:
            P = poly[:, :-1]
            b = poly[:, -1]
            if not np.all(np.dot(P, state) + b <= 0.0):
                # We are not starting in this polytope so we can skip it
                continue
            # Generate the safety constraints
            F = []
            G = []
            h = []
            for j in range(1, self.horizon + 1):
                F.append([None] * (j + 1))
                G.append([None] * (j + 1))
                h.append([None] * (j + 1))
                F[j-1][j] = P
                G[j-1][j] = np.zeros((b.shape[0], u_dim))
                h[j-1][j] = b
                for t in range(j - 1, -1, -1):
                    # At each time step, we need to propogate the previous
                    # constraint backwards (see Google Doc) and add a new
                    # constraint. The new constraint is P x_t + b <= 0
                    F[j-1][t] = np.dot(F[j-1][t+1], A)
                    G[j-1][t] = np.dot(F[j-1][t+1], B)
                    # \eps is an interval so abs(F) \eps gives the maximum
                    # value of F e for e \in \eps
                    epsmax = np.dot(np.abs(F[j-1][t+1]), eps)
                    h[j-1][t] = np.dot(F[j-1][t+1], c) + h[j-1][t+1] + epsmax
            # Now for an action sequence u_0, ..., u_{H-1}, we have that x_i
            # is safe if
            # F[i][0] x_0 + \sum_{t=0}^{h-1} G[i][t] u_t + h[i][0] <= 0
            # So we need to assert this constraint for all 1 <= i <= H
            mat = np.zeros((self.horizon * P.shape[0] +
                            2 * self.horizon * u_dim,
                            self.horizon * u_dim))
            bias = np.zeros(self.horizon * P.shape[0] +
                            2 * self.horizon * u_dim)
            ind = 0
            step = P.shape[0]
            for j in range(self.horizon):
                G[j] += [np.zeros((P.shape[0], u_dim))] * \
                    (self.horizon - j - 1)
                # G[j] = [np.zeros((P.shape[0], u_dim))] * (self.horizon + 1)
                mat[ind:ind+step, :] = np.concatenate(G[j][:-1], axis=1)
                bias[ind:ind+step] = h[j][0] + np.dot(F[j][0], state)
                # bias[ind:ind+step] = -np.ones(step)
                ind += step

            # Add action bounds
            ind2 = 0
            for j in range(self.horizon):
                mat[ind:ind+u_dim, ind2:ind2+u_dim] = np.eye(u_dim)
                bias[ind:ind+u_dim] = -self.action_space.high
                ind += u_dim
                mat[ind:ind+u_dim, ind2:ind2+u_dim] = -np.eye(u_dim)
                bias[ind:ind+u_dim] = self.action_space.low
                ind += u_dim
                ind2 += u_dim

            # Now we satisfy the constraints whenever
            # mat (u_1 u_2 ... u_H)^T + bias <= 0
            # Our objective is || u* - u_0 ||^2 = (u* - u_0)^T (u* - u_0)
            # = u*T u* - 2 u*^T u_0 + u_0^T u_0
            # Since u*^T u* is constant we can leave it out
            # That means we want P to be [[I 0] [0 0]] the objective has a 0.5
            # coefficient on u^T P u, so we use q = -u* rather than q = -2 u^*
            # rather than adding a factor of 2 to P.
            P = 1e-4 * np.eye(self.horizon * u_dim)
            P[:u_dim, :u_dim] = np.eye(u_dim)
            P = cvxopt.matrix(P)
            q = -np.concatenate((action, np.zeros((self.horizon - 1) * u_dim)))
            q = cvxopt.matrix(q)
            G = cvxopt.matrix(mat)
            h = cvxopt.matrix(-bias)
            try:
                sol = cvxopt.solvers.qp(P, q, G, h)
            except Exception:
                # This seems to happen when the primal problem is infeasible
                # sometimes
                sol = {'status': 'infeasible'}

            if sol['status'] != 'optimal':
                # Infeasible or unsolvable problem
                continue
            u0 = np.asarray(sol['x'][:u_dim]).squeeze()
            if len(u0.shape) == 0:
                # Squeeze breaks one-dimensional actions
                u0 = u0[None]
            score = np.linalg.norm(u0 - action)
            if score < best_score:
                best_score = score
                best_u0 = u0

        if best_u0 is None:
            best_u0 = self.backup(state)

        self.saved_state = state
        self.saved_action = best_u0
        return best_u0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if self.saved_state is not None and \
                np.allclose(state, self.saved_state):
            return self.saved_action
        return self.solve(state)

    def unsafe(self,
               state: np.ndarray,
               action: np.ndarray) -> bool:
        res = self.solve(state, action=action)
        return not np.allclose(res, action)


class Shield:
    """
    Construct a shield from a neural policy and a safety layer.
    """

    def __init__(
            self,
            shield_policy: ProjectionPolicy,
            unsafe_policy: SACPolicy):
        self.shield = shield_policy
        self.agent = unsafe_policy
        self.shield_times = 0
        self.agent_times = 0

    def __call__(self, state: np.ndarray, **kwargs) -> np.ndarray:
        proposed_action = self.agent(state, **kwargs)
        if self.shield.unsafe(state, proposed_action):
            act = self.shield(state)
            self.shield_times += 1
            return act
        self.agent_times += 1
        return proposed_action

    def report(self) -> Tuple[int, int]:
        return self.shield_times, self.agent_times

    def reset_count(self):
        self.shield_times = 0
        self.agent_times = 0


class CSCShield:
    """
    Construct a shield from a neural policy and a conservative safety critic.
    """

    def __init__(self, policy: SACPolicy, cost_model: torch.nn.Module,
                 threshold: float = 0.2):
        self.policy = policy
        self.cost_model = cost_model
        self.threshold = threshold

    def __call__(self, state: np.ndarray, **kwargs) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(self.policy(state, **kwargs),
                              dtype=torch.float32)
        iters = 0
        best_action = action
        score = self.cost_model(torch.cat((state, action)))
        best_score = score
        while score > self.threshold and iters < 100:
            action = torch.tensor(self.policy(state, **kwargs),
                                  dtype=torch.float32)
            score = self.cost_model(torch.cat((state, action)))
            if score < best_score:
                best_score = score
                best_action = action
            iters += 1
        return best_action.detach().numpy()

    def report(self) -> Tuple[int, int]:
        return 0, 0

    def reset_count(self):
        pass
