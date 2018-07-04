"""Reinforcement learning experiment code."""


def run_episodes(env, agent, num_episodes):
    """Print out a run of an Agent in an Environment.

    Arguments:
        env (Environment): The environment.
        agent (Agent): The agent.
        num_episodes (int): The number of episodes to run.

    Returns:
        List[float]: The returns of each episode.
    """
    returns = []
    for _ in range(num_episodes):
        env.start_new_episode()
        agent.start_new_episode()
        episodic_return = 0
        reward = None
        step = 0
        obs = env.get_observation()
        actions = env.get_actions()
        while not env.end_of_episode():
            action = agent.act(obs, actions)
            reward = env.react(action)
            obs = env.get_observation()
            agent.observe_reward(obs, reward)
            actions = env.get_actions()
            episodic_return += reward
            step += 1
        returns.append(episodic_return)
    return returns
