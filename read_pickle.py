import pickle
import numpy as np

data = pickle.load(open('PP-1-paths-1000-[0]-end-flag.p', 'rb'))

num_trajectory = 0

# average_length = 0

last_index = 0

num_length = []
for i in range(data.shape[0]):
	if data[i, -1] == 1.0:

		# print(i)

		num_trajectory += 1
		length = i - last_index + 1
		num_length.append(length)

		last_index = i + 1

num_length = np.array(num_length)

print("mean:", num_length.mean(), 
	  "variance:", num_length.var(), 
	  "max:",num_length.max(), 
	  "min:",num_length.min())

print(num_trajectory)
print(data.shape)
print("")