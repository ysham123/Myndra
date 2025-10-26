import numpy as np
import time
from pettingzoo.mpe import simple_spread_v3

class MyndraEnvWrapper:
    """
    Wrapper around PettingZoo MPE environments (e.g., simple_spread_v3)
    to provide a consistent interface for Myndra MARL experiments.
    """

    def __init__(self, env_name="simple_spread_v3", max_cycles=25):
        # For now, we only support simple_spread_v3
        if env_name != "simple_spread_v3":
            raise ValueError(f"Unsupported env: {env_name}")

        # Create the environment in parallel mode
        self.env = simple_spread_v3.parallel_env(max_cycles=max_cycles, continuous_actions=False)
        self.agents = self.env.possible_agents
        self.episode_rewards = {agent: 0.0 for agent in self.agents}
        self.steps = 0

    def reset(self, seed=None):
        """Reset the environment and return the initial observations."""
        obs, info = self.env.reset(seed=seed)
        self.episode_rewards = {agent: 0.0 for agent in self.agents}
        self.steps = 0
        return obs

    def step(self, actions):
        """
        Take one environment step with a dict of agent actions.
        Returns obs, rewards, dones, infos — all dicts keyed by agent.
        """
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        dones = {agent: terms[agent] or truncs[agent] for agent in self.agents}

        # Update episode stats
        for agent, r in rewards.items():
            self.episode_rewards[agent] += r
        self.steps += 1

        return obs, rewards, dones, infos

    def sample_action(self, agent):
        """Return a random valid action for a given agent (for testing)."""
        return self.env.action_space(agent).sample()

    def render(self):
        """Optional visualization (for debugging)."""
        try:
            self.env.render()
        except Exception:
            pass

    def close(self):
        """Clean up resources."""
        self.env.close()

    def get_stats(self):
        """Return episode summary for logging."""
        return {
            "steps": self.steps,
            "total_rewards": dict(self.episode_rewards)
        }