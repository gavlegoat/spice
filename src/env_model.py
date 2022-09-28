from pyearth import Earth
from pyearth._basis import ConstantBasisFunction, LinearBasisFunction, \
    HingeBasisFunction
from typing import Optional, List, Callable
import numpy as np
import torch
import scipy.stats


class ResidualEnvModel(torch.nn.Module):
    """
    A neural model for the residuals leftover by the symbolic model.

    Specifically, for a symbolic model M, a the ResidualEnvModel N should be
    trained such that M(s, a) + N(s, a) gives a complete model of the
    environment.
    """

    def __init__(
            self,
            arch: List[int],
            input_means: np.ndarray,
            input_stds: np.ndarray,
            output_means: np.ndarray,
            output_stds: np.ndarray):
        """
        Initialize a residual environment model.

        The architecuture of the model is specified as a list of layer sizes
        including input and output layers. The network will always be fully
        connected.

        Parameters:
        arch - The archtecture of the model.
        """
        super().__init__()

        layers = []
        for i in range(1, len(arch)):
            layers.append(torch.nn.Linear(arch[i-1], arch[i]))
        self.net = torch.nn.Sequential(*layers)
        self.inp_means = torch.nn.Parameter(
            torch.tensor(input_means, dtype=torch.float32),
            requires_grad=False)
        self.inp_stds = torch.nn.Parameter(
            torch.tensor(input_stds, dtype=torch.float32),
            requires_grad=False)
        self.out_means = torch.nn.Parameter(
            torch.tensor(output_means, dtype=torch.float32),
            requires_grad=False)
        self.out_stds = torch.nn.Parameter(
            torch.tensor(output_stds, dtype=torch.float32),
            requires_grad=False)

    def forward(self,
                x: torch.Tensor,
                normalized: bool = False) -> torch.Tensor:
        if not normalized:
            x = (x - self.inp_means) / self.inp_stds
        x = self.net(x)
        if not normalized:
            x = x * self.out_stds + self.out_means
        return x


class MARSComponent:
    """
    One component of a MARS basis.

    A component is a linear function, possibly with a hinge activation. The
    function is parameterized by optional values term (int), knot (float),
    and negate (bool). Specifically, the interpretation of a component is
    - h(x) = 1 if term = None
    - h(x) = x_term if knot is None
    - h(x) = max(0, x_term - knot) if negate is False
    - h(x) = max(0, knot - x_term) if negate is True
    """

    def __init__(
            self,
            term: Optional[int] = None,
            knot: Optional[float] = None,
            negate: bool = False):
        """
        Initialize a MARS component.

        Parameters:
        term - The variable involved in this component (if any).
        knot - The breakpoint of this component (if any).
        negate - The direction of the knot.
        """
        self.term = term
        self.knot = knot
        self.negate = negate

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate this component at some input.

        Parameters:
        x (1D array) - The point to evaluate this component at.

        Returns:
        The value of this basis function at the given input.
        """
        if self.term is None:
            return 1
        if self.knot is None:
            return x[self.term]
        if self.negate:
            return max(0, self.knot - x[self.term])
        return max(0, x[self.term] - self.knot)

    def __str__(self) -> str:
        """
        Get a string representation of this component.
        """
        if self.term is None:
            return "1.0"
        if self.knot is None:
            return "x" + str(self.term)
        if self.negate:
            return "h({} - x{})".format(self.knot, self.term)
        return "h(x{} - {})".format(self.term, self.knot)

    def get_row(self, x: np.ndarray) -> np.ndarray:
        """
        See MARSModel.get_matrix_at_point.
        """
        res = np.zeros(x.shape[0] + 1)
        if self.term is None:
            res[x.shape[0]] = 1.0
        elif self.knot is None:
            res[self.term] = 1.0
        elif self.negate:
            if self.knot - x[self.term] < 0:
                return res
            res[self.term] = -1.0
            res[x.shape[0]] = self.knot
        else:
            if x[self.term] - self.knot <= 0:
                return res
            res[self.term] = 1.0
            res[x.shape[0]] = -self.knot
        return res


class MARSModel:
    """
    A piecewise-linear model learned by MARS.

    Note that while MARS can work with nonlinear terms, this class is limited
    to piecewise linear models, i.e., sums of constants, linear terms, and
    degree-1 hinge terms. The model is represented as a vector B of basis
    functions and a matrix C of coefficients. At an input x the value of the
    model is given by C B(x)
    """

    def __init__(
            self,
            basis: List[MARSComponent],
            coeffs: np.ndarray,
            err: float,
            input_means: np.ndarray,
            input_stds: np.ndarray,
            output_means: np.ndarray,
            output_stds: np.ndarray):
        """
        Initialize a MARS model.

        Parameters:
        basis - The basis of this MARS model.
        coeffs - The coefficient matrix for this model.
        """
        self.basis = basis
        self.coeffs = coeffs
        self.error = err   # Error in the normalized space
        self.inp_means = input_means
        self.inp_stds = input_stds
        self.out_means = output_means
        self.out_stds = output_stds

    def __call__(self, x: np.ndarray, normalized: bool = False) -> np.ndarray:
        """
        Evaluate the MARS model at a given input.

        Parameters:
        x (1D array) - The input to evaluate at.
        normalized (bool) - If true, assume the input is already normalized

        Returns:
        An estimated output for the given input.
        """
        if not normalized:
            x = (x - self.inp_means) / self.inp_stds
        b = np.array(list(map(lambda f: f(x), self.basis)))
        y = np.dot(self.coeffs, b)
        if normalized:
            return y
        return y * self.out_stds + self.out_means

    def get_matrix_at_point(    # noqa: C901
            self,
            x: np.ndarray,
            s_dim: int,
            steps: int = 1) -> np.ndarray:
        """
        Get the linear model at a particular point.

        This should return a matrix M s.t. self(x) = M (x 1)^T.

        The model is C (h_1(x) h_2(x) ... h_n(x))^T where each h_i has one of
        four forms:
            h_i(x) = 1,
            h_i(x) = x[j],
            h_i(x) = ReLU(x[j] - a), or
            h_i(x) = ReLU(a - x[j])

        We want to find H s.t. H (x 1)^T = (h_1(x) ... h_n(x))^T.
        Then we can find each row H_i of H by cases:
            h_i(x) = 1                -> H_i = e_{n+1}
            h_i(x) = x[j]             -> H_i = e_j
            h_i(x) = ReLU(x[j] - a):
                if x[j] >= a          -> H_i = e_j - a e_{n+1}
                if x[j] <  a          -> H_i = 0
            h_i(x) = ReLU(a - x[j]):
                if x[j] <= a          -> H_i = a e_{n+1} - e_j
                if x[j] >  a          -> H_i = 0

        Note that we also need to deal with normalization here. We have to
        normalize the input before doing the above computation in order for the
        knots in the basis functions to be correct. Then this
        produces a matrix M such that M (x' 1)^T = y' for x'
        and y' normalized according to the stored input and output statistics.
        Specifically, x' = (x - mu_i) / sig_i and y' = (y - mu_o) / sig_o.
        For convenience let M (x 1)^T = M x + b. Then we have
        (y - m_y) / s_y = M ((x - m_x) / s_x) + b + e
        y = (M ((x - m_x) / s_x) + b + e) .* s_y + m_y where .* is element-wise
        y = ((M c/ s_x) (x - m_x) + b + e) .* s_y + m_y where c/ divides
              columns of M by elements of s_x.
        y = ((M c/ s_x) x - (M c/ s_x) m_x + b + e) .* s_y + m_y
        y = (M c/ s_x) x .* s_y - (M c/ s_x) m_x .* s_y + b .* s_y +
              e .* s_y + m_y
        y = (M c/ s_x r* s_y) x + (b .* s_y + m_y - (M c/ s_x) m_x .* s_y) +
              e .* s_y
              where r* means multiply rows of M by elements of s_y
        """
        def get_matrix_help(sa):
            H = np.stack(list(map(lambda f: f.get_row(x), self.basis)))
            M = np.dot(self.coeffs, H)
            tmpM = M[:, :-1] / self.inp_stds
            newM = tmpM * self.out_stds[:, None]
            newb = M[:, -1] * self.out_stds + self.out_means - \
                np.dot(tmpM, self.inp_means) * self.out_stds
            return np.concatenate((newM, newb[:, None]), axis=1)
        normx = (x - self.inp_means) / self.inp_stds
        # Init ret is the approximation at x
        init_ret = get_matrix_help(normx)
        eps = self.error * self.out_stds
        # Handle action disjunctions
        # First, identify base components with have a hinge in an action
        # variable
        action_base = filter(lambda f: f.knot is not None and
                             f.term is not None and f.term >= s_dim,
                             self.basis)
        # Find breakpoints in the action space
        bs = list(map(lambda f: (f.term, f.knot), action_base))
        breaks = [list(set(map(lambda x: x[1],
                               filter(lambda p: p[0] == i, bs))))
                  for i in range(s_dim, x.shape[0])]
        for b in breaks:
            b.sort()
        # Now breaks[i] holds a list of breakpoints for action dim i
        # Look for the action values which give the maximum and minimum slopes
        # based on the current state
        u_dim = x.shape[0] - s_dim
        state = normx[:s_dim]
        inds = [0] * u_dim
        max_act_diff = np.zeros(init_ret.shape)
        while inds[-1] <= len(breaks[-1]):
            # Generate any action in the given piece of the model
            act = np.zeros(u_dim)
            for j in range(u_dim):
                if len(breaks[j]) == 0:
                    act[j] = normx[u_dim + j]
                elif inds[j] == len(breaks[j]):
                    act[j] = breaks[j][-1] + 1.0
                elif inds[j] == 0:
                    act[j] = breaks[j][0] - 1.0
                else:
                    act[j] = (breaks[j][inds[j]] +
                              breaks[j][inds[j] - 1]) / 2.0
            # Get the matrix at the generated action
            M = get_matrix_help(np.concatenate((state, act)))
            max_act_diff = np.maximum(np.abs(M - init_ret), max_act_diff)
            i = 0
            done = False
            while inds[i] >= len(breaks[i]):
                inds[i] = 0
                i += 1
                if i >= len(breaks):
                    done = True
                    break
            if done:
                break
            inds[i] += 1
        eps += np.abs(np.dot(max_act_diff, np.concatenate((x, np.array([1])))))
        # State may step into a new piece
        # To check, unroll abstractly under init_ret and eps and see if we
        # get out of the current piece. If so, add the adjacent piece as
        # error.
        unnormed_state = x[:s_dim]
        abs_state = (unnormed_state, unnormed_state)
        act = x[s_dim:]
        max_diff = np.zeros_like(init_ret)
        for step in range(1, steps):
            low = np.where(init_ret < 0, abs_state[1], abs_state[0])
            high = np.where(init_ret < 0, abs_state[0], abs_state[1])
            nlow = np.sum(init_ret * low, axis=0)
            nhigh = np.sum(init_ret * high, axis=0)
            tmp = np.concatenate((x, np.array([1])))
            abs_state = (nlow - eps - np.abs(np.dot(max_diff, tmp)),
                         nhigh + eps + np.abs(np.dot(max_diff, tmp)))
            # Check whether the new abstract state steps outside the current
            # piece of the state space. Since piece are axis-aligned, we can do
            # this by checking the corners of the abstract state
            corners = [False] * s_dim
            while True:
                st = np.zeros(s_dim)
                for c in range(s_dim):
                    if corners[c]:
                        st[c] = abs_state[1][c]
                    else:
                        st[c] = abs_state[0][c]
                sa = np.concatenate((st, act))
                norm = (sa - self.inp_means) / self.inp_stds
                M = get_matrix_help(norm)
                max_diff = np.maximum(np.abs(M - init_ret), max_diff)
                i = 0
                done = False
                while corners[i]:
                    corners[i] = False
                    i += 1
                    if i >= len(corners):
                        done = True
                        break
                if done:
                    break
                corners[i] = True
        return init_ret, \
            eps + np.abs(np.dot(max_diff, np.concatenate((x, np.array([1])))))

    def __str__(self) -> str:
        """
        Get a string representation of this model.
        """
        ret = "Basis:"
        for fn in self.basis:
            ret += "\n" + str(fn)
        ret += "\nCoeffs:\n" + str(self.coeffs.tolist())
        return ret


class EnvModel:
    """
    A full environment model including a symbolic model and a neural model.

    This model includes a symbolic (MARS) model of the dynamics, a neural
    model which accounts for dynamics not captured by the symbolic model, and a
    second neural model for the reward function.
    """

    def __init__(
            self,
            mars: MARSModel,
            symb_reward: MARSModel,
            net: ResidualEnvModel,
            reward: ResidualEnvModel,
            use_neural_model: bool):
        """
        Initialize an environment model.

        Parameters:
        mars - A symbolic model.
        net - A neural model for the residuals.
        reward - A neural model for the reward.
        """
        self.mars = mars
        self.symb_reward = symb_reward
        self.net = net
        self.reward = reward
        self.use_neural_model = use_neural_model

    def __call__(self,
                 state: np.ndarray,
                 action: np.ndarray,
                 use_neural_model: bool = True) -> np.ndarray:
        """
        Predict a new state and reward value for a given state-action pair.

        Parameters:
        state (1D array) - The current state of the system.
        action (1D array) - The action to take

        Returns:
        A tuple consisting of the new state and the reward.
        """
        inp = np.concatenate((state, action), axis=0)
        symb = self.mars(inp)
        if self.use_neural_model:
            neur = self.net(torch.tensor(inp, dtype=torch.float32)). \
                detach().numpy()
            rew = self.reward(torch.tensor(inp, dtype=torch.float32)).item()
        else:
            neur = np.zeros_like(symb)
            rew = self.symb_reward(inp)[0]
        return symb + neur, rew

    def get_symbolic_model(self) -> MARSModel:
        """
        Get the symbolic component of this model.
        """
        return self.mars

    def get_residual_model(self) -> ResidualEnvModel:
        """
        Get the residual neural component of this model.
        """
        return self.net

    def get_confidence(self) -> float:
        return self.confidence

    @property
    def error(self) -> float:
        return self.mars.error


def get_environment_model(     # noqa: C901
        input_states: np.ndarray,
        actions: np.ndarray,
        output_states: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        lows: torch.Tensor,
        highs: torch.Tensor,
        seed: int = 0,
        use_neural_model: bool = True,
        arch: Optional[List[int]] = None,
        cost_model: torch.nn.Module = None,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        data_stddev: float = 0.01,
        model_pieces: int = 10) -> EnvModel:
    """
    Get a neurosymbolic model of the environment.

    This function takes a dataset consisting of M sample input states and
    actions together with the observed output states and rewards. It then
    trains a neurosymbolic model to imitate that data.

    An architecture may also be supplied for the neural parts of the model.
    The architecture format is a list of hidden layer sizes. The networks are
    always fully-connected. The default architecture is [280, 240, 200].

    Parameters:
    input_states (M x S array) - An array of input states.
    actions (M x A array) - An array of actions taken from the input states.
    output_states (M x S array) - The measured output states.
    rewards (M array) - The measured rewards.
    arch: A neural architecture for the residual and reward models.
    """

    states_mean = np.concatenate((input_states, output_states),
                                 axis=0).mean(axis=0)
    states_std = np.maximum(np.concatenate((input_states, output_states),
                                           axis=0).std(axis=0), 1e-5)
    actions_mean = actions.mean(axis=0)
    actions_std = np.maximum(actions.std(axis=0), 1e-5)
    rewards_mean = rewards.mean()
    rewards_std = np.maximum(rewards.std(), 1e-5)

    print("State stats:", states_mean, states_std)
    print("Action stats:", actions_mean, actions_std)
    print("Reward stats:", rewards_mean, rewards_std)

    input_states = (input_states - states_mean) / states_std
    output_states = (output_states - states_mean) / states_std
    actions = (actions - actions_mean) / actions_std
    rewards = (rewards - rewards_mean) / rewards_std

    if policy is not None:
        policy_actions = (actions - actions_mean) / actions_std
        next_policy_actions = (actions - actions_mean) / actions_std

    X = np.concatenate((input_states, actions), axis=1)
    Y = output_states

    terms = 20
    # Lower penalties allow more model complexity.
    symb = Earth(max_degree=1, max_terms=model_pieces, penalty=1.0,
                 endspan=terms, minspan=terms)
    symb.fit(X, Y)

    coeffs = symb.coef_
    basis = []
    for (i, fn) in enumerate(symb.basis_):
        if fn.is_pruned():
            continue
        if isinstance(fn, ConstantBasisFunction):
            comp = MARSComponent()
        elif isinstance(fn, LinearBasisFunction):
            comp = MARSComponent(fn.get_variable())
        elif isinstance(fn, HingeBasisFunction):
            comp = MARSComponent(term=fn.get_variable(),
                                 knot=fn.get_knot(),
                                 negate=fn.get_reverse())
        else:
            raise Exception("Unrecognized basis function: " + type(fn))
        basis.append(comp)

    parsed_mars = MARSModel(
        basis, coeffs, 0.0,
        np.concatenate((states_mean, actions_mean)),
        np.concatenate((states_std, actions_std)),
        states_mean, states_std)

    # Convert the MARS model to our own representation.
    if np.any(np.abs(coeffs) >= 1e3):
        print("Coefficients are exploding")
        with open("debug_dump" + str(seed), 'w') as dump:
            dump.write("Model:\n")
            dump.write(str(parsed_mars))
            dump.write("\n")
            dump.write("Data statistics:\n")
            dump.write("State means: ")
            dump.write(str(states_mean))
            dump.write("\n")
            dump.write("State stds: ")
            dump.write(str(states_std))
            dump.write("\n")
            dump.write("Action means: ")
            dump.write(str(actions_mean))
            dump.write("\n")
            dump.write("Action std: ")
            dump.write(str(actions_std))
            dump.write("\n")
            dump.write("Reward means: ")
            dump.write(str(rewards_mean))
            dump.write("\n")
            dump.write("Reward std: ")
            dump.write(str(rewards_std))
            dump.write("\n")
            dump.write("Normalized data leading to an exploding model:\n")
            dump.write("Input, Action, Output, Reward\n")
            for (inp, out, act, rew) in \
                    zip(input_states, output_states, actions, rewards):
                dump.write(str((inp, act, out, rew)))
                dump.write("\n")
            input_states = input_states * states_std + states_mean
            output_states = output_states * states_std + states_mean
            actions = actions * actions_std + actions_mean
            rewards = rewards * rewards_std + rewards_mean
            dump.write("Raw data leading to an exploding model:\n")
            dump.write("Input, Action, Output, Reward\n")
            for (inp, out, act, rew) in \
                    zip(input_states, output_states, actions, rewards):
                dump.write(str((inp, act, out, rew)))
                dump.write("\n")
        raise RuntimeError("Coefficients are exploding")

    Yh = np.array([parsed_mars(state, normalized=True) for state in X])

    # Get the maximum distance between a predction and a datapoint
    diff = np.amax(np.abs(Yh - output_states))

    # Get a confidence interval based on the quantile of the chi-squared
    # distribution
    conf = data_stddev * np.sqrt(scipy.stats.chi2.ppf(
        0.9, output_states.shape[1]))
    err = diff + conf
    print("Computed error:", err, "(", diff, conf, ")")
    parsed_mars.error = err

    # Set up a neural network for the residuals.
    state_action = np.concatenate((input_states, actions), axis=1)
    if arch is None:
        arch = [280, 240, 200]
    arch.insert(0, state_action.shape[1])
    arch.append(output_states.shape[1])
    model = ResidualEnvModel(
        arch,
        np.concatenate((states_mean, actions_mean)),
        np.concatenate((states_std, actions_std)),
        states_mean, states_std)
    model.train()

    # Set up a training environment
    optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss = torch.nn.MSELoss()

    data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(state_action, dtype=torch.float32),
                torch.tensor(output_states - Yh, dtype=torch.float32)),
            batch_size=128,
            shuffle=True)

    # Train the neural network.
    for epoch in range(100):
        losses = []
        for batch_data, batch_outp in data:
            pred = model(batch_data, normalized=True)
            # Normalize predictions and labels to the range [-1, 1]
            loss_val = loss(pred, batch_outp)
            losses.append(loss_val.item())
            optim.zero_grad()
            loss_val.backward()
            optim.step()
        print("Epoch:", epoch,
              torch.tensor(losses, dtype=torch.float32).mean())

    model.eval()

    # Get a symbolic reward model
    reward_symb = Earth(max_degree=1, max_terms=model_pieces, penalty=1.0,
                        endspan=terms, minspan=terms)
    reward_symb.fit(X, rewards)

    rew_coeffs = reward_symb.coef_
    rew_basis = []
    for fn in reward_symb.basis_:
        if fn.is_pruned():
            continue
        if isinstance(fn, ConstantBasisFunction):
            rew_basis.append(MARSComponent())
        elif isinstance(fn, LinearBasisFunction):
            rew_basis.append(MARSComponent(fn.get_variable()))
        elif isinstance(fn, HingeBasisFunction):
            rew_basis.append(MARSComponent(term=fn.get_variable(),
                                           knot=fn.get_knot(),
                                           negate=fn.get_reverse()))
        else:
            raise Exception("Unrecognized basis function: " + type(fn))
    parsed_rew = MARSModel(
        rew_basis, rew_coeffs, 0.01,
        np.concatenate((states_mean, actions_mean)),
        np.concatenate((states_std, actions_std)),
        rewards_mean[None], rewards_std[None])

    # Set up a neural network for the rewards
    arch[-1] = 1
    rew_model = ResidualEnvModel(
        arch,
        np.concatenate((states_mean, actions_mean)),
        np.concatenate((states_std, actions_std)),
        rewards_mean[None], rewards_std[None])

    optim = torch.optim.Adam(rew_model.parameters(), lr=1e-5)
    loss = torch.nn.SmoothL1Loss()

    # Set up training data for the rewards
    reward_data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(state_action, dtype=torch.float32),
                torch.tensor(rewards[:, None], dtype=torch.float32)),
            batch_size=128,
            shuffle=True)

    rew_model.train()

    # Train the network.
    for epoch in range(100):
        losses = []
        for batch_data, batch_outp in reward_data:
            pred = rew_model(batch_data, normalized=True)
            loss_val = loss(pred, batch_outp)
            losses.append(loss_val.item())
            optim.zero_grad()
            loss_val.backward()
            optim.step()
        print("Epoch:", epoch,
              torch.tensor(losses, dtype=torch.float32).mean())

    rew_model.eval()

    if policy is not None:
        if cost_model is None:
            cost_model = ResidualEnvModel(
                arch,
                np.concatenate((states_mean, actions_mean)),
                np.concatenate((states_std, actions_std)),
                0.0, 1.0)

        optim = torch.optim.Adam(cost_model.parameters(), lr=1e-4)
        loss = torch.nn.SmoothL1Loss()

        # Set up training data for the cost_model
        cost_data = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(input_states, dtype=torch.float32),
                    torch.tensor(actions, dtype=torch.float32),
                    torch.tensor(policy_actions, dtype=torch.float32),
                    torch.tensor(next_policy_actions, dtype=torch.float32),
                    torch.tensor(costs[:, None], dtype=torch.float32)),
                batch_size=128,
                shuffle=True)

        cost_model.train()

        # Negative weight overestimates the safety critic rather than
        # underestimating
        q_weight = -1.0
        for epoch in range(1):
            losses = []
            for batch_states, batch_acts, batch_pacts, \
                    batch_npacts, batch_costs in cost_data:
                pred = cost_model(torch.cat((batch_states, batch_acts), dim=1))
                main_loss = loss(pred, batch_costs)
                q_cur = cost_model(torch.cat((batch_states, batch_pacts),
                                             dim=1))
                q_next = cost_model(torch.cat((batch_states, batch_npacts),
                                              dim=1))
                q_cat = torch.cat([q_cur, q_next], dim=1)
                q_loss = torch.logsumexp(q_cat, dim=1).mean() * q_weight
                q_loss = q_loss - pred.mean() * q_weight
                loss_val = main_loss + q_loss
                losses.append(loss_val.item())
                optim.zero_grad()
                loss_val.backward()
                optim.step()
            print("Epoch:", epoch,
                  torch.tensor(losses, dtype=torch.float32).mean())

        cost_model.eval()

    print(symb.summary())
    print(parsed_mars)
    print("Model MSE:", np.mean(np.sum((Yh - output_states)**2, axis=1)))
    print(reward_symb.summary())

    return EnvModel(parsed_mars, parsed_rew, model, rew_model,
                    use_neural_model), cost_model
