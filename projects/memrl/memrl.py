import sys
from collections import namedtuple
from os.path import dirname, realpath
from statistics import mean

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable=wrong-import-position
from research.rl_agents import TabularQLearningAgent, epsilon_greedy

from envs import RandomMaze, memory_architecture


def trace_episode(env, agent, num_episodes, min_return=-500, pause=False):
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
            '''
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
            '''
            reward = env.react(action)
            agent.observe_reward(env.get_observation(), reward)
            episodic_return += reward
            step += 1
        print(f'EPISODE RETURN: {episodic_return}')
        print()


def evaluate_agent(env, agent, num_episodes, min_return=-500):
    returns = []
    for _ in range(num_episodes):
        env.start_new_episode()
        agent.start_new_episode()
        episodic_return = 0
        while not env.end_of_episode() and episodic_return > min_return:
            action = agent.act(
                observation=env.get_observation(),
                actions=env.get_actions(),
            )
            reward = env.react(action)
            episodic_return += reward
        returns.append(episodic_return)
    return mean(returns)


def train_agent(env, agent, num_episodes, min_return=-500):
    for _ in range(num_episodes):
        env.start_new_episode()
        agent.start_new_episode()
        episodic_return = 0
        while not env.end_of_episode() and episodic_return > min_return:
            action = agent.act(
                observation=env.get_observation(),
                actions=env.get_actions(),
            )
            reward = env.react(action)
            agent.observe_reward(env.get_observation(), reward)
            episodic_return += reward


def train_and_evaluate(env, agent, num_episodes, eval_frequency=10, test_num_episodes=10, min_return=-500):
    for epoch_num in range(int(num_episodes / eval_frequency)):
        train_agent(env, agent, eval_frequency)
        mean_return = evaluate_agent(env, agent, test_num_episodes, min_return=min_return)
        yield mean_return


def trace_experiment(params, pause=True):
    env = memory_architecture(RandomMaze)(
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


def run_experiment(params):
    env = memory_architecture(RandomMaze)(
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
        test_num_episodes=params.test_num_episodes,
        min_return=params.min_return,
    )


def dict_replace(orig, **kwargs):
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
        'test_num_episodes',
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
    'test_num_episodes': 10,
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
    episodes = range(0, NUM_EPISODES, EVAL_FREQENCY)
    experiments = [1, 2, 3, 4]
    #experiments = [4]
    exp_params = [globals()[f'EXP_{i}_PARAM'] for i in experiments]
    exp_results = [run_experiment(param) for param in exp_params]
    for episode, *exps in zip(episodes, *exp_results):
        print(episode, *exps)


if __name__ == '__main__':
    main()
    #trace_experiment(EXP_4_PARAM, pause=False)
