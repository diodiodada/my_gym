import pickle
import numpy as np

def get_length():

	data = pickle.load(open('PP-1-paths-1000-[0, 1, 2, 3]-top4.p', 'rb'))

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


def get_data():

    data = pickle.load(open('PP-24-paths-2400-2.p', 'rb'))

    # data normalization
    x = np.array([0])
    y = x + 1
    z = x + 2

    data_x = data[:, x]
    data_y = data[:, y]
    data_z = data[:, z]

    data_x_min = data_x.min()
    data_x_max = data_x.max()
    data_y_min = data_y.min()
    data_y_max = data_y.max()
    data_z_min = data_z.min()
    data_z_max = data_z.max()

    print(data_x.min())
    print(data_x.max())
    print(data_y.min())
    print(data_y.max())
    print(data_z.min())
    print(data_z.max())

    return 


get_length()