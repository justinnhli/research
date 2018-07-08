"""Driver for memory-RL experiments."""

import sys
from collections import namedtuple
from os.path import dirname, realpath
from statistics import mean

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.rl_agents import Agent, TabularQLearningAgent, epsilon_greedy
from research.rl_environments import Action

from envs import RandomMaze, memory_architecture


def trace_episode(env, agent, num_episodes, min_return=-500, pause=False):
    """Run some episodes and print out what's going on.

    Arguments:
        env (Environment): The environment.
        agent (Agent): The agent.
        num_episodes (int): The number of episodes to run.
        min_return (float): A minimum threshold under which to stop an episode.
        pause (bool): Whether to pause after each step.
    """
    for episode_num in range(num_episodes):
        print(f'EPISODE {episode_num}')
        env.start_new_episode()
        agent.start_new_episode()
        episodic_return = 0
        step = 1
        while not env.end_of_episode() and episodic_return > min_return:
            action = agent.act(
                observation=env.get_observation(),
                actions=env.get_actions(),
            )
            print(f'Step {step}')
            agent.print_value_function()
            print(f'Observation: {env.get_observation()}')
            print(f'Actions:')
            for possible_action in env.get_actions():
                print(f'    {possible_action}')
            print(f'Action: {action}')
            print()
            if pause:
                input('<enter>')
            reward = env.react(action)
            agent.observe_reward(env.get_observation(), reward)
            episodic_return += reward
            step += 1
        print(f'EPISODE RETURN: {episodic_return}')
        print()


def run_episodes(env, agent, num_episodes, min_return=-500, update_agent=True, new_episode_hook=None):
    """Run some episodes and return the mean return.

    Arguments:
        env (Environment): The environment.
        agent (Agent): The agent.
        num_episodes (int): The number of episodes to run.
        min_return (float): A minimum threshold under which to stop an episode.
        update_agent (bool): Whether the agent will observe rewards
        new_episode_hook (Function[Environment, Agent]): A hook at each new episode.

    Returns:
        float: The mean return over all episodes.
    """
    returns = []
    for _ in range(num_episodes):
        env.start_new_episode()
        agent.start_new_episode()
        episodic_return = 0
        if new_episode_hook is not None:
            new_episode_hook(env, agent)
        while not env.end_of_episode() and episodic_return > min_return:
            action = agent.act(
                observation=env.get_observation(),
                actions=env.get_actions(),
            )
            reward = env.react(action)
            if update_agent:
                agent.observe_reward(env.get_observation(), reward)
            episodic_return += reward
        returns.append(episodic_return)
    return mean(returns)


def evaluate_agent(env, agent, num_episodes, min_return=-500, new_episode_hook=None):
    """Evaluate an agent.

    Arguments:
        env (Environment): The environment.
        agent (Agent): The agent.
        num_episodes (int): The number of episodes to run.
        min_return (float): A minimum threshold under which to stop an episode.
        new_episode_hook (Function[Environment, Agent]): A hook at each new episode.

    Returns:
        float: The mean return over all episodes.
    """
    return run_episodes(
        env,
        agent,
        num_episodes,
        update_agent=False,
        min_return=min_return,
        new_episode_hook=new_episode_hook,
    )


def train_agent(env, agent, num_episodes, min_return=-500, new_episode_hook=None):
    """Train an agent.

    Arguments:
        env (Environment): The environment.
        agent (Agent): The agent.
        num_episodes (int): The number of episodes to run.
        min_return (float): A minimum threshold under which to stop an episode.
        new_episode_hook (Function[Environment, Agent]): A hook at each new episode.
    """
    run_episodes(
        env,
        agent,
        num_episodes,
        update_agent=True,
        min_return=min_return,
        new_episode_hook=new_episode_hook,
    )


def train_and_evaluate(env, agent, num_episodes, **kwargs):
    """Train an agent and evaluate it at regular intervals.

    Arguments:
        env (Environment): The environment.
        agent (Agent): The agent.
        num_episodes (int): The number of episodes to run.
        eval_frequency (int): The number of episodes between each evaluation
        eval_num_episodes (int): The number of episodes to run for evaluation.
        min_return (float): A minimum threshold under which to stop an episode.
        new_episode_hook (Function[Environment, Agent]): A hook at each new episode.

    Yields:
        float: The mean return of each evaluation.
    """
    # pylint: disable = differing-param-doc, differing-type-doc, missing-doc-param
    eval_frequency = kwargs.get('eval_frequency', 10)
    eval_num_episodes = kwargs.get('eval_num_episodes', 10)
    min_return = kwargs.get('min_return', -500)
    new_episode_hook = kwargs.get('new_episode_hook', None)
    for epoch_num in range(int(num_episodes / eval_frequency)):
        train_agent(env, agent, eval_frequency, new_episode_hook=new_episode_hook)
        mean_return = evaluate_agent(
            env,
            agent,
            eval_num_episodes,
            min_return=min_return,
            new_episode_hook=new_episode_hook
        )
        yield mean_return


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
        load_goal_path=params.load_goal_path,
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
        'load_goal_path',
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
    'load_goal_path': False,
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
    load_goal_path=True,
))

EXP_4_PARAM = ExperimentParameter(**dict_replace(
    PARAMETER_DEFAULTS,
    randomize=True,
    representation='coords',
    load_goal_path=True,
    map_representation='coords',
))


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
    main()
    #trace_experiment(EXP_4_PARAM, pause=False)
