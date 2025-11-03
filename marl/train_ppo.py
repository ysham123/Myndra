import time
import csv
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical
import numpy as np
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marl.env_wrapper import MyndraEnvWrapper
from systems.profiler import Profiler

class StubPlanner:
    """Lightweight deterministic planner that outputs context vectors."""
    def __init__(self, context_dim=4, interval=32, cache=True, profiler=None):
        self.context_dim = context_dim
        self.interval = interval
        self.cache = cache
        self.profiler = profiler
        self._last_ctx = np.zeros(context_dim, dtype=np.float32)
    
    def maybe_update(self, step_count):
        """Return a context vector. Update every `interval` steps."""
        if step_count % self.interval != 0 and self.cache:
            return self._last_ctx
        
        t0 = time.time()
        # Example: cycle through 4 simple "goal" modes
        mode = (step_count // self.interval) % self.context_dim
        ctx = np.zeros(self.context_dim, dtype=np.float32)
        ctx[mode] = 1.0
        self._last_ctx = ctx
        if self.profiler:
            self.profiler.log_metric("planner_latency_ms", (time.time() - t0) * 1000.0)
        return ctx

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, lr=3e-4):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
    
    def act(self, obs):
        #given an observation, sample an action and return log prob
        obs = torch.tensor(obs, dtype=torch.float32)
        probs = self.actor(obs)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, buffer, gamma=0.99, clip_eps=0.2, epochs=4, use_amp=False):
        # Convert buffer data to numpy first, then to tensors
        import numpy as np
        data = list(zip(*buffer.storage))
        obs = torch.tensor(np.array(data[0]), dtype=torch.float32)
        actions = torch.tensor(np.array(data[1]), dtype=torch.int64)
        rewards = torch.tensor(np.array(data[2]), dtype=torch.float32)
        next_obs = torch.tensor(np.array(data[3]), dtype=torch.float32)
        dones = torch.tensor(np.array(data[4]), dtype=torch.float32)
        old_log_probs = torch.tensor(np.array(data[5]), dtype=torch.float32)

        # Compute advantages once, detached from computation graph
        with torch.no_grad():
            values = self.critic(obs).squeeze()
            next_values = self.critic(next_obs).squeeze()
            targets = rewards + gamma * next_values * (1-dones)
            advantages = targets - values
            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Set up AMP scaler
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
        
        for _ in range(epochs):
            with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available()):
                probs = self.actor(obs)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions)

                ratio = torch.exp(new_log_probs - old_log_probs)

                clip_adv = torch.clamp(ratio, 1-clip_eps, 1 + clip_eps) * advantages 
                loss_actor = -torch.min(ratio * advantages, clip_adv).mean()

                value_pred = self.critic(obs).squeeze()
                loss_critic = F.mse_loss(value_pred, targets)

                loss = loss_actor + 0.5 * loss_critic

            self.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

class RolloutBuffer:
    def __init__(self):
        self.storage = []
    
    def add(self, obs, action, reward, next_obs, done, log_prob):
        self.storage.append((obs, action, reward, next_obs, done, log_prob))
    
    def clear(self):
        self.storage.clear()

#training loop

def train(env_name="simple_spread_v3", total_steps=5000, log_interval=1000, seed=None,
          use_planner=False, planner_interval=32, context_dim=4, planner_cache=True, method="ippo",
          actors=1, use_amp=False, use_compile=False, target_return=None):
    # Create multiple environments for parallel rollouts
    envs = [MyndraEnvWrapper(env_name) for _ in range(actors)]
    profiler = Profiler()

    # Deterministic seeding for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    # Set up planner if enabled
    planner = None
    if use_planner:
        planner = StubPlanner(
            context_dim=context_dim,
            interval=planner_interval,
            cache=planner_cache,
            profiler=profiler
        )
        # Initialize all environments with zero context
        for env in envs:
            env.set_planner_context(np.zeros(context_dim, dtype=np.float32))

    # Structured output paths for multi-seed runs
    if seed is not None:
        out_dir = Path("results/marl") / env_name / method / f"seed_{seed}"
    else:
        out_dir = Path("results/marl") / env_name / method
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics.csv"
    profile_path = out_dir / "train_profile.json"

    start_time = time.time()
    episode_rewards = []
    time_to_target = None  # Will be set when target is reached
    target_reached = False
    
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_reward", "steps_per_second", "elapsed_sec", 
                        "planner_on", "planner_interval", "context_dim", "actors", "amp", "compile"])

    # Initialize observations for all actor environments
    obs_list = [env.reset() for env in envs]
    # Use obs_size_with_ctx if planner is enabled
    obs_dim = envs[0].obs_size_with_ctx if use_planner else envs[0].obs_size
    agent = PPOAgent(obs_dim=obs_dim, act_dim=envs[0].act_size)
    
    # Apply torch.compile if enabled
    if use_compile:
        agent.actor = torch.compile(agent.actor)
        agent.critic = torch.compile(agent.critic)
    
    buffer = RolloutBuffer()

    step = 0

    while step < total_steps:
        profiler.start("rollout")
        
        # Update planner context for all envs if enabled
        if use_planner:
            ctx = planner.maybe_update(step)
            for env in envs:
                env.set_planner_context(ctx)
        
        # Collect experience from all actor environments
        for i, env in enumerate(envs):
            obs = obs_list[i]
            
            # Check if we need to reset
            if not obs or len(obs) == 0:
                obs = env.reset()
                obs_list[i] = obs
            
            # Collection experience - only act for agents with observations
            active_agents = [a for a in env.agents if a in obs]
            if not active_agents:
                obs = env.reset()
                obs_list[i] = obs
                active_agents = [a for a in env.agents if a in obs]
            
            action_log_probs = {a: agent.act(obs[a]) for a in active_agents}
            actions = {a: action_log_probs[a][0] for a in active_agents}
            log_probs = {a: action_log_probs[a][1] for a in active_agents}
            
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Track average reward
            if rewards:
                avg_reward = sum(rewards.values()) / len(rewards)
                episode_rewards.append(avg_reward)
            
            # Add experiences to buffer
            for a in active_agents:
                if a in rewards and a in next_obs:
                    buffer.add(obs[a], actions[a], rewards[a], next_obs[a], dones[a], log_probs[a])
            
            obs_list[i] = next_obs
        
        # Increment step by number of actors (parallel collection)
        step += actors

        profiler.stop("rollout")

        #occasionally update PPO

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_second = step / elapsed if elapsed > 0 else 0
            mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
            
            # Check if target return is reached
            if target_return is not None and not target_reached and mean_reward >= target_return:
                time_to_target = elapsed
                target_reached = True
                profiler.log_metric("time_to_target_return_sec", time_to_target)
                print(f"  Target return {target_return} reached at step {step} ({time_to_target:.2f}s)")

            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, mean_reward, steps_per_second, round(elapsed, 2),
                               int(use_planner), planner_interval if use_planner else 0,
                               context_dim if use_planner else 0, actors, int(use_amp), int(use_compile)])

            episode_rewards.clear()

            profiler.start("update")
            agent.update(buffer, use_amp=use_amp)
            profiler.stop("update")
            
            # Sample GPU utilization after update
            gpu_util = profiler.sample_gpu_util()
            if gpu_util is not None:
                profiler.log_metric("gpu_util_percent", gpu_util)
            
            buffer.clear()
            print(f"{step} steps collected, updating PPO...")
    # Close all environments
    for env in envs:
        env.close()
    profiler.save(profile_path)

    # Free resources (important for multi-seed runs)
    del envs, agent, buffer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("---Training Complete---")

    return {
        "metrics_csv": str(metrics_path),
        "profile_json": str(profile_path)
    }

if __name__ == "__main__":
    train()