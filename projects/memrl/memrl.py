"""Driver for memory-RL experiments."""

import sys
from collections import namedtuple
from os.path import dirname, realpath

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.rl_core import trace_episode, train_agent, evaluate_agent, train_and_evaluate
from research.rl_agents import Agent, TabularQLearningAgent, epsilon_greedy
from research.rl_environments import Action

from envs import RandomMaze, memory_architecture


def trace_experiment(params, pause=False):
    """Run an experiment and print out what's going on.

    Arguments:
        params (ExperimentParameter): The parameters of the experiment.
        pause (bool): Whether to pause after each step.
    """
    env = create_env(params)
    agent = epsilon_greedy(TabularQLearningAgent)(
        # Tabular Q Learning Agent
        random_seed=params.agent_random_seed,
        learning_rate=params.learning_rate,
        discount_rate=params.discount_rate,
        # Epsilon Greedy
        exploration_rate=params.exploration_rate,
    )
    trace_episode(
        env,
        agent,
        num_episodes=params.num_episodes,
        min_return=params.min_return,
        pause=pause,
    )


def create_env(params):
    """Create an environment based on experiment parameters.

    Arguments:
        params (ExperimentParameter): The parameters of the experiment.

    Returns:
        Environment: The environment as described by the parameters.
    """
    return memory_architecture(RandomMaze)(
        # Random Maze
        random_seed=params.env_random_seed,
        size=params.size,
        randomize=params.randomize,
        representation=params.representation,
        # Memory Architecture
        explicit_actions=params.explicit_actions,
        map_representation=params.map_representation,
    )


def run_experiment(params):
    """Run an experiment.

    Arguments:
        params (ExperimentParameter): The parameters of the experiment.

    Returns:
        Sequence[float]: The mean returns of each evaluation.
    """
    env = create_env(params)
    agent = epsilon_greedy(TabularQLearningAgent)(
        # Tabular Q Learning Agent
        random_seed=params.agent_random_seed,
        learning_rate=params.learning_rate,
        discount_rate=params.discount_rate,
        # Epsilon Greedy
        exploration_rate=params.exploration_rate,
    )
    return train_and_evaluate(
        env,
        agent,
        num_episodes=params.num_episodes,
        eval_frequency=params.eval_frequency,
        eval_num_episodes=params.eval_num_episodes,
        min_return=params.min_return,
        new_episode_hook=(lambda env, agent: load_ltm(env, agent, params)),
    )


def run_optimal_agent(params):
    """Evaluate the optimal agent for an experiment.

    Arguments:
        params (ExperimentParameter): The parameters of the experiment.

    Returns:
        float: The mean return of the evaluation.
    """
    env = create_env(params)
    agent = OptimalAgent()
    return evaluate_agent(
        env,
        agent,
        num_episodes=params.eval_num_episodes,
        min_return=params.min_return,
        new_episode_hook=(lambda env, agent: load_ltm(env, agent, params)),
    )


def load_ltm(env, _, params):
    """Load RandomMaze paths into a MemoryArchitecture agent.

    Arguments:
        env (MemoryArchitecture): The environment.
        _ (Agent): The agent.
        params (ExperimentParameter): The parameters of the experiment.
    """
    env.ltm = set()
    if params.map_representation == 'symbol':
        LocDir = namedtuple('LocationDirection', ['location', 'direction'])
        for location, direction in env.goal_map.items():
            env.ltm.add(LocDir(location, direction))
    else:
        LocDir = namedtuple('LocationDirection', ['row', 'col', 'direction'])
        for location, direction in env.goal_map.items():
            env.ltm.add(LocDir(
                location // env.size,
                location % env.size,
                direction,
            ))


def dict_replace(orig, **kwargs):
    """Copy a dictionary with some values replaced.

    Arguments:
        orig (dict): The original dictionary.
        **kwargs (Object): Key-values to be replaced.

    Returns:
        dict: The new dictionary
    """
    result = dict(**orig)
    result.update(**kwargs)
    return result


class OptimalAgent(Agent):
    """The optimal agent for random mazes."""

    # pylint: disable = abstract-method

    def _retrieved_matches(self, observation):
        retrieved = self.retrieved_attrs(observation)
        if not retrieved:
            return False
        match = all(
            (
                f'retrieval_{attr}' in observation
                and observation[f'retrieval_{attr}'] == observation[f'perceptual_{attr}']
            ) for attr in self.perceptual_attrs(observation)
        )
        return match

    def buffer_attrs(self, observation, buf):
        """Get the attributes in a buffer.

        Arguments:
            observation (Observation): The observation.
            buf (str): The buffer to get attributes from

        Returns:
            Set[str]: The attributes in that buffer.
        """
        # pylint: disable = no-self-use
        return set(key[len(buf) + 1:] for key in observation if key.startswith(buf + '_'))

    def perceptual_attrs(self, observation):
        """Get the attributes the perceptual buffer.

        Arguments:
            observation (Observation): The observation.

        Returns:
            Set[str]: The attributes in the perceptual buffer.
        """
        return self.buffer_attrs(observation, 'perceptual')

    def retrieved_attrs(self, observation):
        """Get the attributes the retrieved buffer.

        Arguments:
            observation (Observation): The observation.

        Returns:
            Set[str]: The attributes in the retrieved buffer.
        """
        return self.buffer_attrs(observation, 'retrieval')

    def query_attrs(self, observation):
        """Get the attributes the query buffer.

        Arguments:
            observation (Observation): The observation.

        Returns:
            Set[str]: The attributes in the query buffer.
        """
        return self.buffer_attrs(observation, 'query')

    def act(self, observation, actions):
        """Update the value function and decide on the next action.

        In order, check for
            * retrieved == perceptual
            * retrieved != perceptual, incorrect query
            * partial/no query

        Arguments:
            observation (State): The observation of the environment.
            actions (list[Action]): List of available actions.

        Returns:
            Action: The action the agent takes.
        """
        action = None
        perceptual_attrs = self.perceptual_attrs(observation)
        query_attrs = self.query_attrs(observation)
        if self._retrieved_matches(observation) and 'retrieval_direction' in observation:
            # if retrieved, output action
            action = Action(
                'copy',
                src_buf='retrieval',
                src_attr='direction',
                dst_buf='action',
                dst_attr='name',
            )
        elif query_attrs - perceptual_attrs:
            # if query is incorrect
            attr = min(query_attrs - perceptual_attrs)
            action = Action('delete', buf='perceptual', attr=min(query_attrs - perceptual_attrs))
        else:
            # no query or query is subset; retrieve for location
            # FIXME deal with representation change
            attr = min(perceptual_attrs - query_attrs)
            action = Action(
                'copy',
                src_buf='perceptual',
                src_attr=attr,
                dst_buf='query',
                dst_attr=attr,
            )
        assert action is not None, 'No action generated\n' + str(observation)
        assert action in actions, 'Action unrecognized\n' + str(action)
        return action

    def observe_reward(self, observation, reward): # noqa: D102
        pass


ExperimentParameter = namedtuple(
    'ExperimentParameter',
    [
        # ENVIRONMENT PARAMETERS
        # Random Maze
        'env_random_seed',
        'size',
        'randomize',
        'representation',
        # Memory Architecture
        'explicit_actions',
        'map_representation',
        # AGENT PARAMETERS
        # Tabular Q Learning Agent
        'agent_random_seed',
        'learning_rate',
        'discount_rate',
        # Epsilon Greedy
        'exploration_rate',
        # EVALUATION PARAMETERS
        'num_episodes',
        'eval_frequency',
        'eval_num_episodes',
        'min_return',
    ],
)

SIZE = 5
NUM_EPISODES = 3000
EVAL_FREQENCY = 100

PARAMETER_DEFAULTS = {
    # ENVIRONMENT PARAMETERS
    # Random Maze
    'env_random_seed': 8675309,
    'size': SIZE,
    'randomize': False,
    'representation': 'symbol',
    # Memory Architecture
    'explicit_actions': False,
    'map_representation': 'symbol',
    # AGENT PARAMETERS
    # Tabular Q Learning Agent
    'agent_random_seed': 8675309,
    'learning_rate': 0.1,
    'discount_rate': 0.9,
    # Epsilon Greedy
    'exploration_rate': 0.1,
    # EVALUATION PARAMETERS
    'num_episodes': NUM_EPISODES,
    'eval_frequency': EVAL_FREQENCY,
    'eval_num_episodes': 1000,
    'min_return': -5000,
}

EXP_1_PARAM = ExperimentParameter(**dict_replace(
    PARAMETER_DEFAULTS,
    explicit_actions=True,
))

EXP_2_PARAM = ExperimentParameter(**dict_replace(
    PARAMETER_DEFAULTS,
    randomize=True,
    explicit_actions=True,
))

EXP_3_PARAM = ExperimentParameter(**dict_replace(
    PARAMETER_DEFAULTS,
    randomize=True,
))

EXP_4_PARAM = ExperimentParameter(**dict_replace(
    PARAMETER_DEFAULTS,
    randomize=True,
    representation='coords',
    map_representation='coords',
))


def show_optimal_rewards():
    """Calculate the optimal rewards."""
    experiments = [3, 4]
    for exp in experiments:
        print(exp, run_optimal_agent(globals()[f'EXP_{exp}_PARAM']))


def main():
    """Run experiments."""
    episodes = range(0, NUM_EPISODES, EVAL_FREQENCY)
    experiments = [1, 2, 3, 4]
    experiments = [4]
    exp_params = [globals()[f'EXP_{i}_PARAM'] for i in experiments]
    exp_results = [run_experiment(param) for param in exp_params]
    for episode, *exps in zip(episodes, *exp_results):
        print(episode, *exps)


if __name__ == '__main__':
    #show_optimal_rewards()
    main()
    #trace_experiment(EXP_4_PARAM, pause=False)
