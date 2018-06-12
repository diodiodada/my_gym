import pickle
import numpy as np

data = pickle.load(open('PP-24-paths-2400-0.p', 'rb'))
data_1 = pickle.load(open('PP-24-paths-2400-1.p', 'rb'))
data_2 = pickle.load(open('PP-24-paths-2400-2.p', 'rb'))
data_3 = pickle.load(open('PP-24-paths-2400-3.p', 'rb'))
data_4 = pickle.load(open('PP-24-paths-2400-4.p', 'rb'))
data_5 = pickle.load(open('PP-24-paths-2400-5.p', 'rb'))
data_6 = pickle.load(open('PP-24-paths-2400-6.p', 'rb'))
data_7 = pickle.load(open('PP-24-paths-2400-7.p', 'rb'))
data_8 = pickle.load(open('PP-24-paths-2400-8.p', 'rb'))
data_9 = pickle.load(open('PP-24-paths-2400-9.p', 'rb'))

data = np.concatenate((data, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9), axis = 0)

pickle.dump(data, open("PP-24-paths-24000.p", "wb"))