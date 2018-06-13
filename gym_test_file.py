import gym
env = gym.make('FetchPickAndPlace-v0')

for j in range(100):
	env.reset()
	for _ in range(10):
	    env.render()
	    # obs, _, _, _ = env.step(env.action_space.sample()) # take a random action
	    obs, _, _, _ = env.step([0, 0, 0, 0])
	    print(obs)