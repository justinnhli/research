"""Reinforcement learning experiment code."""

from statistics import mean

from .rl_agents import Agent


def run_episodes(env, agent, num_episodes, min_return=-500, update_agent=True, new_episode_hook=None):
    """Run some episodes and return the mean return.

    Arguments:
        env (Environment): The environment.
        agent (Agent): The agent.
        num_episodes (int): The number of episodes to run.
        min_return (float): A minimum threshold under which to stop an episode.
        update_agent (bool): Whether the agent will observe rewards
        new_episode_hook (Callable[[Environment, Agent], None]): A hook at each new episode.

    Returns:
        float: The mean return over all episodes.
    """
    returns = []
    for _ in range(num_episodes):
        env.start_new_episode()
        agent.start_new_episode()
        if new_episode_hook is not None:
            new_episode_hook(env, agent)
        episodic_return = 0
        step = 0
        while not env.end_of_episode() and episodic_return > min_return:
            action = agent.act(
                observation=env.get_observation(),
                actions=env.get_actions(),
            )
            reward = env.react(action)
            if update_agent:
                agent.observe_reward(env.get_observation(), reward, actions=env.get_actions())
            episodic_return += reward
            step += 1
        returns.append(episodic_return)
    return mean(returns)


def evaluate_agent(env, agent, num_episodes, min_return=-500, new_episode_hook=None):
    """Evaluate an agent.

    Arguments:
        env (Environment): The environment.
        agent (Agent): The agent.
        num_episodes (int): The number of episodes to run.
        min_return (float): A minimum threshold under which to stop an episode.
        new_episode_hook (Callable[Environment, Agent]): A hook at each new episode.

    Returns:
        float: The mean return over all episodes.
    """
    class ExploitAgent(Agent):
        """An Agent that only selects the best action."""

        # pylint: disable = abstract-method

        def __init__(self, agent):
            """Initialize the ExploitAgent.

            Arguments:
                agent (Agent): The underlying agent.

            """
            super().__init__()
            self.agent = agent

        def act(self, observation, actions): # noqa: D102
            action = self.agent.get_best_stored_action(observation, actions=actions)
            if action is None:
                return self.rng.choice(actions)
            else:
                return action

    return run_episodes(
        env,
        ExploitAgent(agent),
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
        new_episode_hook (Callable[[Environment, Agent], None]): A hook at each new episode.
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
        new_episode_hook (Callable[[Environment, Agent], None]): A hook at each new episode.

    Yields:
        float: The mean return of each evaluation.
    """
    # pylint: disable = differing-param-doc, differing-type-doc, missing-param-doc
    eval_frequency = kwargs.get('eval_frequency', 10)
    eval_num_episodes = kwargs.get('eval_num_episodes', 10)
    if eval_frequency == 0:
        train_episodes = num_episodes
    else:
        train_episodes = eval_frequency
    min_return = kwargs.get('min_return', -500)
    new_episode_hook = kwargs.get('new_episode_hook', None)
    for episode_num in range(0, num_episodes, train_episodes):
        train_agent(env, agent, train_episodes, new_episode_hook=new_episode_hook)
        should_evaluate = (
            eval_frequency == 0 or
            (eval_frequency > 0 and episode_num % eval_frequency == 0)
        )
        if should_evaluate:
            mean_return = evaluate_agent(
                env,
                agent,
                eval_num_episodes,
                min_return=min_return,
                new_episode_hook=new_episode_hook
            )
            yield mean_return
