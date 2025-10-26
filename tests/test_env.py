from marl.env_wrapper import MyndraEnvWrapper

env = MyndraEnvWrapper()
obs = env.reset()

for step in range(5):
    actions = {a:env.sample_action(a) for a in env.agents}
    obs, rewards, dones, infos = env.step(actions)
    print(f"Step{step}: rewards{rewards}")

env.close()