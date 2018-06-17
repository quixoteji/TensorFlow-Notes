import numpy as np
birth_file = open('./data/lowbwt.dat').read()
# birth_data = birth_file.split('\n')



# ax = [x for x in a if x.isdigit()]
# print(ax)

birth_data = birth_file.split('\n')[2:]


dataset = []
for record in birth_data:
    data = [x for x in record.split(' ') if x.isdigit()]
    dataset.append(data)

dataset = np.asarray(dataset[:-3])
print(dataset.shape)
x_vals = np.asarray([x[2:9] for x in dataset])
y_vals = np.asarray([x[1] for x in dataset])
print(x_vals.shape)
print(y_vals.shape)

