import gym
env = gym.make('FetchPickAndPlace-v0')
env.reset()
for _ in range(1000):
    env.render()
    obs, _, _, _ = env.step(env.action_space.sample()) # take a random action
    # print(obs)