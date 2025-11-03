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

    def update(self, buffer, gamma=0.99, clip_eps=0.2, epochs=4):
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

        for _ in range(epochs):
            probs = self.actor(obs)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            clip_adv = torch.clamp(ratio, 1-clip_eps, 1 + clip_eps) * advantages 
            loss_actor = -torch.min(ratio * advantages, clip_adv).mean()

            value_pred = self.critic(obs).squeeze()
            loss_critic = F.mse_loss(value_pred, targets)

            loss = loss_actor + 0.5 * loss_critic

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class RolloutBuffer:
    def __init__(self):
        self.storage = []
    
    def add(self, obs, action, reward, next_obs, done, log_prob):
        self.storage.append((obs, action, reward, next_obs, done, log_prob))
    
    def clear(self):
        self.storage.clear()

#training loop

def train(env_name="simple_spread_v3", total_steps=5000, log_interval=1000, seed=None):
    env = MyndraEnvWrapper(env_name)
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

    # Structured output paths for multi-seed runs
    if seed is not None:
        out_dir = Path("results/marl") / env_name / "ippo" / f"seed_{seed}"
    else:
        out_dir = Path("results/marl") / env_name / "ippo"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics.csv"
    profile_path = out_dir / "train_profile.json"

    start_time = time.time()
    episode_rewards = []
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_reward", "steps_per_second", "elapsed_sec"])

    obs = env.reset()
    agent = PPOAgent(obs_dim=env.obs_size, act_dim=env.act_size)
    buffer = RolloutBuffer()

    step = 0

    while step < total_steps:
        profiler.start("rollout")
        
        # Check if we need to reset (no agents have observations)
        if not obs or len(obs) == 0:
            obs = env.reset()
        
        #collection experience - only act for agents with observations
        active_agents = [a for a in env.agents if a in obs]
        if not active_agents:
            obs = env.reset()
            active_agents = [a for a in env.agents if a in obs]
        
        action_log_probs = {a: agent.act(obs[a]) for a in active_agents}
        actions = {a: action_log_probs[a][0] for a in active_agents}  # Extract just the actions
        log_probs = {a: action_log_probs[a][1] for a in active_agents}  # Extract log probs
        
        next_obs, rewards, dones, infos = env.step(actions)
        # track average reward
        if rewards:
            avg_reward = sum(rewards.values()) / len(rewards)
            episode_rewards.append(avg_reward)

        # Only add experiences for agents that have data in this step
        for a in active_agents:
            if a in rewards and a in next_obs:
                buffer.add(obs[a], actions[a], rewards[a], next_obs[a], dones[a], log_probs[a])
        obs = next_obs
        step += 1

        profiler.stop("rollout")

        #occasionally update PPO

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_second = step / elapsed if elapsed > 0 else 0
            mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0

            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, mean_reward, steps_per_second, round(elapsed, 2)])

            episode_rewards.clear()

            profiler.start("update")
            agent.update(buffer)
            profiler.stop("update")
            buffer.clear()
            print(f"{step} steps collected, updating PPO...")
    env.close()
    profiler.save(profile_path)

    # Free resources (important for multi-seed runs)
    del env, agent, buffer
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