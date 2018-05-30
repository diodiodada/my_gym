import pickle
import numpy as np
data = pickle.load(open('Pick-Place-Push-category-10000.p', 'rb'))

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
print("mean:", num_length.mean())
print("variance:", num_length.var(), num_length.max(), num_length.min())

print(num_trajectory)
print(data.shape)