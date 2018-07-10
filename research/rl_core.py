"""Reinforcement learning experiment code."""

from statistics import mean


def trace_episode(env, agent, num_episodes, min_return=-500, pause=False, new_episode_hook=None):
    """Run some episodes and print out what's going on.

    Arguments:
        env (Environment): The environment.
        agent (Agent): The agent.
        num_episodes (int): The number of episodes to run.
        min_return (float): A minimum threshold under which to stop an episode.
        pause (bool): Whether to pause after each step.
        new_episode_hook (Function[Environment, Agent]): A hook at each new episode.
    """
    for episode_num in range(num_episodes):
        print(f'EPISODE {episode_num}')
        env.start_new_episode()
        agent.start_new_episode()
        if new_episode_hook is not None:
            new_episode_hook(env, agent)
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
    # pylint: disable = differing-param-doc, differing-type-doc, missing-param-doc
    eval_frequency = kwargs.get('eval_frequency', 10)
    eval_num_episodes = kwargs.get('eval_num_episodes', 10)
    min_return = kwargs.get('min_return', -500)
    new_episode_hook = kwargs.get('new_episode_hook', None)
    for _ in range(int(num_episodes / eval_frequency)):
        train_agent(env, agent, eval_frequency, new_episode_hook=new_episode_hook)
        mean_return = evaluate_agent(
            env,
            agent,
            eval_num_episodes,
            min_return=min_return,
            new_episode_hook=new_episode_hook
        )
        yield mean_return
