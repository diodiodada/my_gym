import gym
env = gym.make('CarRacing-v0')
observation = env.reset()

for j in range(50000):
	# print()
	# for _ in range(5):
    env.render()
    obs, _, _, _ = env.step(env.action_space.sample()) # take a random action
    # obs, _, _, _ = env.step([0, 0, 0, 0])
    # print(obs["my_new_observation"][0:3])