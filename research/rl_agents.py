"""Reinforcement learning agents."""

from collections import defaultdict
from typing import Any, Optional, Type, Iterable, Mapping, Callable, Hashable, Dict

from .randommixin import RandomMixin
from .rl_environments import State, Action


class Agent(RandomMixin):
    """A reinforcement learning agent."""

    def __init__(self, **kwargs):
        # type: (*Any, **Any) -> None
        """Initialize the Agent.

        Arguments:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.prev_observation = None # type: Optional[State]
        self.prev_action = None # type: Optional[Action]

    def start_new_episode(self):
        # type: () -> None
        """Prepare the agent for a new episode."""
        self.prev_observation = None
        self.prev_action = None

    def _get_value(self, observation, action):
        # type: (State, Action) -> float
        """Get the Q value for an action at an observation.

        Arguments:
            observation (State): The observation
            action (Action): The action

        Returns:
            float: The value for the action at the observation.
        """
        raise NotImplementedError()

    def _get_stored_actions(self, observation):
        # type: (State) -> Iterable[Action]
        """Get all actions with stored values at an observation.

        Arguments:
            observation (State): The observation.

        Returns:
            Iterable[Action]: The stored actions at the observation.
        """
        raise NotImplementedError()

    def _get_best_stored_action(self, observation, actions=None):
        # type: (State, Optional[Iterable[Action]]) -> Action
        """Get the action with the highest value at an observation.

        Arguments:
            observation (State): The observation.
            actions (Iterable[Action]): The available actions. Defaults to None.

        Returns:
            Action: The best action for the given observation.
        """
        if actions is None:
            actions = self._get_stored_actions(observation)
        if not actions:
            return None
        else:
            return max(actions, key=(lambda action: self._get_value(observation, action)))

    def _get_best_stored_value(self, observation, actions=None):
        # type: (State, Optional[Iterable[Action]]) -> float
        """Get the highest value at an observation.

        Arguments:
            observation (State): The observation.
            actions (Iterable[Action]): The available actions. Defaults to None.

        Returns:
            float: The value of the best action for the given observation.
        """
        return self._get_value(observation, self._get_best_stored_action(observation, actions=actions))

    def best_act(self, observation, actions=None):
        # type: (State, Optional[Iterable[Action]]) -> Action
        """Take the action with the highest value.

        Arguments:
            observation (State): The observation of the environment.
            actions (Optional[Iterable[Action]]): List of available actions.

        Returns:
            Action: The best action.
        """
        best_action = self._get_best_stored_action(observation, actions=actions)
        if best_action is None:
            best_action = self.rng.choice(actions)
        return self.force_act(observation, best_action)

    def act(self, observation, actions):
        # type: (State, Iterable[Action]) -> Action
        """Take the appropriate action.

        Arguments:
            observation (State): The observation of the environment.
            actions (Iterable[Action]): List of available actions.

        Returns:
            Action: The action the agent takes.
        """
        return self.best_act(observation, actions=actions)

    def force_act(self, observation, action):
        # type: (State, Action) -> Action
        """Take the specified action.

        Arguments:
            observation (State): The observation of the environment.
            action (Action): The action to take.

        Returns:
            Action: The specified action.
        """
        self.prev_observation = observation
        if observation is None:
            self.prev_action = None
        else:
            self.prev_action = action
        return action

    def observe_reward(self, observation, reward, actions=None):
        # type: (State, float, Optional[Iterable[Action]]) -> None
        """Update the value function with the reward.

        Arguments:
            observation (State): The current observation.
            reward (float): The reward from the previous action.
            actions (Iterable[Action]): The available actions. Defaults to None.
        """
        raise NotImplementedError()

    def print_value_function(self):
        # type: () -> None
        """Print the value function."""
        raise NotImplementedError()


class TabularQLearningAgent(Agent):
    """A tabular Q-learning reinforcement learning agent."""

    def __init__(self, learning_rate, discount_rate, **kwargs):
        # type: (float, float, **Any) -> None
        """Initialize a tabular Q-learning agent.

        Arguments:
            learning_rate (float): The learning rate (alpha).
            discount_rate (float): The discount rate (gamma).
            **kwargs: Arbitrary keyword arguments.
        """
        self.value_function = defaultdict((lambda: defaultdict(float))) # type: Dict[State, Dict[Action, float]]
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        super().__init__(**kwargs)

    def _get_value(self, observation, action): # noqa: D102
        # type: (State, Action) -> float
        return self.value_function.get(observation, {}).get(action, 0)

    def _get_stored_actions(self, observation): # noqa: D102
        # type: (State) -> Iterable[Action]
        if observation not in self.value_function:
            return []
        return self.value_function[observation].keys()

    def observe_reward(self, observation, reward, actions=None): # noqa: D102
        # type: (State, float, Optional[Iterable[Action]]) -> None
        if self.prev_observation is None or self.prev_action is None:
            return
        prev_value = self._get_value(self.prev_observation, self.prev_action)
        next_value = reward + self.discount_rate * self._get_best_stored_value(observation)
        new_value = (1 - self.learning_rate) * prev_value + self.learning_rate * next_value
        self.value_function[self.prev_observation][self.prev_action] = new_value

    def print_value_function(self): # noqa: D102
        # type: () -> None
        for observation, values in sorted(self.value_function.items(), key=(lambda kv: str(kv[0]))):
            print(observation)
            for action, value in sorted(values.items(), key=(lambda kv: kv[1]), reverse=True):
                print('    {}: {:.3f}'.format(action, value))

    def print_policy(self):
        # type: () -> None
        """Print the policy."""
        for observation in sorted(self.value_function.keys(), key=str):
            print(observation)
            best_action = self._get_best_stored_action(observation)
            print('    {}: {:.3f}'.format(best_action, self._get_value(observation, best_action)))


class LinearQLearner(Agent):
    """A Q learning with linear value function approximation."""

    def __init__(self, learning_rate, discount_rate, **kwargs):
        # type: (float, float, Callable[[State, Optional[Action]], Mapping[Hashable, float]], **Any) -> None
        """Initialize a tabular Q-learning agent.

        Arguments:
            learning_rate (float): The learning rate (alpha).
            discount_rate (float): The discount rate (gamma).
            **kwargs: Arbitrary keyword arguments.
        """
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        super().__init__(**kwargs)
        self.weights = defaultdict(lambda: defaultdict(float)) # type: Dict[Action, Dict[Hashable, float]]

    def _get_value(self, observation, action): # noqa: D102
        # type: (State, Action) -> float
        weights = self.weights[action]
        return sum(
            weights[feature] * value for feature, value
            in observation.items()
        )

    def _get_stored_actions(self, observation): # noqa: D102
        # type: (State) -> Iterable[Action]
        return self.weights.keys()

    def observe_reward(self, observation, reward, actions=None): # noqa: D102
        # type: (State, float, Optional[Iterable[Action]]) -> None
        if self.prev_observation is None or self.prev_action is None:
            return
        prev_value = self._get_value(self.prev_observation, self.prev_action)
        next_value = reward + self.discount_rate * self._get_best_stored_value(observation, actions=actions)
        diff = next_value - prev_value
        for feature, value in self.prev_observation.items():
            weight = self.weights[self.prev_action][feature]
            self.weights[self.prev_action][feature] = weight + (self.learning_rate * diff) * value
            if self.weights[self.prev_action][feature] == 0:
                del self.weights[self.prev_action][feature]

    def print_value_function(self): # noqa: D102
        # type: () -> None
        for action, weights in self.weights.items():
            print(action)
            for feature, weight in weights.items():
                print('   ', feature, weight)


def epsilon_greedy(cls):
    # type: (Type[Agent]) -> Type[Agent]
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

    class EpsilonGreedyMetaAgent(cls): # type: ignore
        """An Agent subclass that behaves epsilon greedily."""

        def __init__(self, exploration_rate, **kwargs): # noqa: D102
            # type: (float, **Any) -> None
            """Initialize the epsilon-greedy agent.

            Arguments:
                exploration_rate (float): The probability of random action.
                **kwargs: Arbitrary keyword arguments.
            """
            self.exploration_rate = exploration_rate
            super().__init__(**kwargs)

        def act(self, observation, actions): # noqa: D102
            # type: (State, Iterable[Action]) -> Action
            # pylint: disable = missing-docstring
            if self.rng.random() < self.exploration_rate:
                return super().force_act(observation, self.rng.choice(actions))
            else:
                return super().best_act(observation, actions)

    return EpsilonGreedyMetaAgent


def feature_transformed(cls):
    # type: (Type[Agent]) -> Type[Agent]
    """Apply a feature transform before the value function.

    Arguments:
        cls (class): The Agent superclass.

    Returns:
        class: An Agent subclass that uses features.
    """
    assert issubclass(cls, Agent)

    class FeatureTransformedMetaAgent(cls): # type: ignore
        """An Agent subclass that uses features."""

        def __init__(self, feature_fn, **kwargs):
            # type: (Callable[[State], State], **Any) -> None
            """Initialize the feature agent.

            Arguments:
                feature_fn (Callable[[State], State]): The feature transformation function.
                **kwargs: Arbitrary keyword arguments.
            """
            self.feature_fn = feature_fn
            super().__init__(**kwargs)

        def best_act(self, observation, actions):
            # type: (State, Iterable[Action]) -> Action
            # pylint: disable = missing-docstring
            return super().best_act(self.feature_fn(observation), actions)

        def act(self, observation, actions):
            # type: (State, Iterable[Action]) -> Action
            # pylint: disable = missing-docstring
            return super().act(self.feature_fn(observation), actions)

        def force_act(self, observation, action):
            # type: (State, Action) -> Action
            # pylint: disable = missing-docstring
            return super().force_act(self.feature_fn(observation), action)

        def observe_reward(self, observation, reward, actions=None): # noqa: D102
            # type: (State, float) -> None
            # pylint: disable = missing-docstring
            return super().observe_reward(self.feature_fn(observation), reward, actions=actions)

    return FeatureTransformedMetaAgent
