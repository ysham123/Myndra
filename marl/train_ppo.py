import time
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical
from marl.env_wrapper import MyndraEnvWrapper
from systems.profiler import Profiler

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def act(self, obs):
        #given an observation, sample an action and return log prob
        obs = torch.tensor(obs, dtype=torch.float32)
        probs = self.actor(obs)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, buffer):
        #todo:compute loss, backprop
        pass 

class RolloutBuffer:
    def __init__(self):
        self.storage = []
    
    def add(self, obs, action, reward, next_obs, done, log_prob):
        self.storage.append((obs, action, reward, next_obs, done, log_prob))
    
    def clear(self):
        self.storage.clear()

#training loop

def train(env_name="simple_spread_v3", total_steps=5000, log_interval=1000):
    env = MyndraEnvWrapper(env_name)
    profiler = Profiler()

    obs = env.reset()
    agent = PPOAgent(obs_dim=env.obs_size, act_dim=env.act_size)
    buffer = RolloutBuffer()

    step = 0

    while step < total_steps:
        profiler.track_start("rollout")
        #collection experience
        actions = {a:agent.act(obs[a]) for a in env.agents}
        next_obs, rewards, dones, infos = env.step(actions)

        for a in env.agents:
            buffer.add(obs[a], actions[a], rewards[a], next_obs[a], dones[a], None)
        obs = next_obs
        step += 1

        profiler.track_end("rollout")

        #occasionally update PPO

        if step % log_interval == 0:
            profiler.track_start("update")
            agent.update(buffer)
            profiler.track_end("update")
            buffer.clear()
            print(f"{step} steps collected, updating PPO...")
    env.close()
    profiler.save("results/marl/train_profile.json")
    print("---Training Complete---")

if __name__ == "__main__":
    train()