import gym
env = gym.make('FPPGC1-v0')

for _ in range(100):
	env.reset()
	for _ in range(5):
	    env.render()
	    obs, _, _, _ = env.step([0, 0, 0, 0]) 
