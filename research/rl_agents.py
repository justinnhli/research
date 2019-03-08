"""Reinforcement learning agents."""

from collections import defaultdict

from .randommixin import RandomMixin


class Agent(RandomMixin):
    """A reinforcement learning agent."""

    def __init__(self, *args, **kwargs):
        """Initialize the Agent.

        Arguments:
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.prev_observation = None
        self.prev_action = None
        self.start_new_episode()

    def start_new_episode(self):
        """Prepare the agent for a new episode."""
        self.prev_observation = None
        self.prev_action = None

    def observe_reward(self, observation, reward, actions=None):
        """Update the value function with the reward.

        Arguments:
            observation (State): The current observation.
            reward (float): The reward from the previous action.
            actions (Seq[Action]): The available actions. Defaults to None.
        """
        raise NotImplementedError()

    def get_value(self, observation, action):
        """Get the Q value for an action at an observation.

        Arguments:
            observation (State): The observation
            action (Action): The action

        Returns:
            float: The value for the action at the observation.
        """
        raise NotImplementedError()

    def get_stored_actions(self, observation):
        """Get all actions with stored values at an observation.

        Arguments:
            observation (State): The observation.

        Returns:
            Sequence[Action]: The stored actions at the observation.
        """
        raise NotImplementedError()

    def get_best_stored_action(self, observation, actions=None):
        """Get the action with the highest value at an observation.

        Arguments:
            observation (State): The observation.
            actions (Seq[Action]): The available actions. Defaults to None.

        Returns:
            Action: The best action for the given observation.
        """
        if actions is None:
            actions = self.get_stored_actions(observation)
        if not actions:
            return None
        else:
            return max(actions, key=(lambda action: self.get_value(observation, action)))

    def get_best_stored_value(self, observation, actions=None):
        """Get the highest value at an observation.

        Arguments:
            observation (State): The observation.
            actions (Seq[Action]): The available actions. Defaults to None.

        Returns:
            float: The value of the best action for the given observation.
        """
        return self.get_value(observation, self.get_best_stored_action(observation, actions=actions))

    def act(self, observation, actions):
        """Update the value function and decide on the next action.

        Arguments:
            observation (State): The observation of the environment.
            actions (Sequence[Action]): List of available actions.

        Returns:
            Action: The action the agent takes.
        """
        best_action = None
        best_value = None
        for action in actions:
            value = self.get_value(observation, action)
            if value is not None and (best_value is None or value > best_value):
                best_action = action
                best_value = value
        if best_action is None:
            best_action = self.rng.choice(actions)
        assert best_action is None or best_action in actions
        return self.force_act(observation, best_action)

    def force_act(self, observation, action):
        """Update the value function and return a specific action.

        Arguments:
            observation (State): The observation of the environment.
            action (Action): The action to return.

        Returns:
            Action: The action the agent takes.
        """
        self.prev_observation = observation
        if observation is None:
            self.prev_action = None
        else:
            self.prev_action = action
        return action

    def print_value_function(self):
        """Print the value function."""
        raise NotImplementedError()


class TabularQLearningAgent(Agent):
    """A tabular Q-learning reinforcement learning agent."""

    def __init__(self, learning_rate, discount_rate, *args, **kwargs):
        """Initialize a tabular Q-learning agent.

        Arguments:
            learning_rate (float): The learning rate (alpha).
            discount_rate (float): The discount rate (gamma).
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.value_function = defaultdict((lambda: defaultdict(float)))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

    def get_value(self, observation, action): # noqa: D102
        return self.value_function.get(observation, {}).get(action, 0)

    def get_stored_actions(self, observation): # noqa: D102
        if observation not in self.value_function:
            return []
        return self.value_function[observation].keys()

    def observe_reward(self, observation, reward, actions=None): # noqa: D102
        if self.prev_observation is None or self.prev_action is None:
            return
        prev_value = self.get_value(self.prev_observation, self.prev_action)
        next_value = reward + self.discount_rate * self.get_best_stored_value(observation)
        new_value = (1 - self.learning_rate) * prev_value + self.learning_rate * next_value
        self.value_function[self.prev_observation][self.prev_action] = new_value

    def print_value_function(self): # noqa: D102
        for observation, values in sorted(self.value_function.items(), key=(lambda kv: str(kv[0]))):
            print(observation)
            for action, value in sorted(values.items(), key=(lambda kv: kv[1]), reverse=True):
                print('    {}: {:.3f}'.format(action, value))

    def print_policy(self):
        """Print the policy."""
        for observation in sorted(self.value_function.keys(), key=str):
            print(observation)
            best_action = self.get_best_stored_action(observation)
            print('    {}: {:.3f}'.format(best_action, self.get_value(observation, best_action)))


class LinearQLearner(Agent):
    """A Q learning with linear value function approximation."""

    def __init__(self, learning_rate, discount_rate, feature_extractor, *args, **kwargs):
        """Initialize a tabular Q-learning agent.

        Arguments:
            learning_rate (float): The learning rate (alpha).
            discount_rate (float): The discount rate (gamma).
            feature_extractor (Callable[[Observation, Optional[Action]], Mapping[Hashable, float]]):
                A function that extracts features from a state.
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.feature_extractor = feature_extractor
        self.weights = defaultdict(lambda: defaultdict(float))

    def get_value(self, observation, action): # noqa: D102
        if action not in self.weights:
            return 0
        weights = self.weights[action]
        return sum(
            weights[feature] * value for feature, value
            in self.feature_extractor(observation, action=action).items()
        )

    def get_stored_actions(self, observation): # noqa: D102
        return self.weights.keys()

    def observe_reward(self, observation, reward, actions=None): # noqa: D102
        if self.prev_observation is None or self.prev_action is None:
            return
        prev_value = self.get_value(self.prev_observation, self.prev_action)
        next_value = reward + self.discount_rate * self.get_best_stored_value(observation, actions=actions)
        diff = next_value - prev_value
        features = self.feature_extractor(self.prev_observation, action=self.prev_action)
        for feature, value in features.items():
            weight = self.weights[self.prev_action][feature]
            self.weights[self.prev_action][feature] = weight + (self.learning_rate * diff) * value
            if self.weights[self.prev_action][feature] == 0:
                del self.weights[self.prev_action][feature]

    def print_value_function(self): # noqa: D102
        for action, weights in self.weights.items():
            print(action)
            for feature, weight in weights.items():
                print('   ', feature, weight)


def epsilon_greedy(cls):
    """Decorate an Agent to be epsilon-greedy.

    This decorator function takes a class (and a value of epsilon) and, on the
    fly, creates a subclass which acts in an epsilon-greedy manner.
    Specifically, it overrides Agent.act() to select a random action with
    epsilon probability.

    Arguments:
        cls (class): The Agent superclass.

    Returns:
        class: An Agent subclass that behaves epsilon greedily.
    """
    assert issubclass(cls, Agent)

    class EpsilonGreedyMetaAgent(cls):
        """An Agent subclass that behaves epsilon greedily."""

        def __init__(self, exploration_rate, *args, **kwargs): # noqa: D102
            """Initialize the epsilon-greedy agent.

            Arguments:
                exploration_rate (float): The probability of random action.
                *args: Arbitrary positional arguments.
                **kwargs: Arbitrary keyword arguments.
            """
            super().__init__(*args, **kwargs)
            self.exploration_rate = exploration_rate

        def act(self, observation, actions): # noqa: D102
            # pylint: disable = missing-docstring
            if self.rng.random() < self.exploration_rate:
                return super().force_act(observation, self.rng.choice(actions))
            else:
                return super().act(observation, actions)

    return EpsilonGreedyMetaAgent


def feature_function(cls):
    """Apply a feature transform before the value function.

    Arguments:
        cls (class): The Agent superclass.

    Returns:
        class: An Agent subclass that uses features.
    """
    assert issubclass(cls, Agent)

    class FeatureMetaAgent(cls):
        """An Agent subclass that uses features."""

        def __init__(self, feature_fn, *args, **kwargs):
            """Initialize the feature agent.

            Arguments:
                feature_fn (Callable[[Observation], Hashable]): The feature transformation function.
                *args: Arbitrary positional arguments.
                **kwargs: Arbitrary keyword arguments.
            """
            super().__init__(*args, **kwargs)
            self.feature_fn = feature_fn

        def get_value(self, observation, action): # noqa: D102
            # pylint: disable = missing-docstring
            return super().get_value(self.feature_fn(observation), action)

        def get_stored_actions(self, observation): # noqa: D102
            # pylint: disable = missing-docstring
            return super().get_stored_actions(self.feature_fn(observation))

        def observe_reward(self, observation, reward): # noqa: D102
            # pylint: disable = missing-docstring
            return super().observe_reward(self.feature_fn(observation), reward)

    return FeatureMetaAgent
