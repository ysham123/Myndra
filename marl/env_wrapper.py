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
        
        # Get observation and action space dimensions from any agent (all agents have same spaces)
        first_agent = self.agents[0]
        self.obs_size = self.env.observation_space(first_agent).shape[0]
        self.act_size = self.env.action_space(first_agent).n
        
        # Planner context (optional, set via set_planner_context)
        self._planner_ctx = None

    def reset(self, seed=None):
        """Reset the environment and return the initial observations."""
        obs, info = self.env.reset(seed=seed)
        self.episode_rewards = {agent: 0.0 for agent in self.agents}
        self.steps = 0
        # Augment observations with planner context if available
        obs = {a: self._augment_obs(obs[a]) for a in obs}
        return obs

    def step(self, actions):
        """
        Take one environment step with a dict of agent actions.
        Returns obs, rewards, dones, infos — all dicts keyed by agent.
        """
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        # Use keys from the returned dicts instead of self.agents
        # since some agents might be removed when done
        dones = {agent: terms.get(agent, False) or truncs.get(agent, False) for agent in self.agents}

        # Update episode stats
        for agent, r in rewards.items():
            self.episode_rewards[agent] += r
        self.steps += 1
        
        # Augment observations with planner context if available
        obs = {a: self._augment_obs(obs[a]) for a in obs}

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

    def set_planner_context(self, ctx_vec):
        """
        ctx_vec: shape (context_dim,), or None to clear.
        Stored and appended to each agent observation.
        """
        if ctx_vec is None:
            self._planner_ctx = None
        else:
            self._planner_ctx = np.asarray(ctx_vec, dtype=np.float32)
    
    @property
    def obs_size_with_ctx(self):
        """Return observation size including planner context if present."""
        base = self.obs_size
        extra = 0 if self._planner_ctx is None else self._planner_ctx.shape[0]
        return base + extra
    
    def _augment_obs(self, obs):
        """Augment observation with planner context if available."""
        if self._planner_ctx is None:
            return obs
        return np.concatenate([obs, self._planner_ctx], axis=-1)

    def get_stats(self):
        """Return episode summary for logging."""
        return {
            "steps": self.steps,
            "total_rewards": dict(self.episode_rewards)
        }