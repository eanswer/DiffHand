import argparse
import os
import numpy as np

parser = argparse.ArgumentParser('')
parser.add_argument('--name', type = str)

args = parser.parse_args()

input_weight_path = os.path.join('./', args.name + '.txt')
output_weight_path = os.path.join('./', args.name + '.npy')

with open(input_weight_path, 'r') as fp:
    data = fp.readline().split()
    n, m = int(data[0]), int(data[1])
    lbs_mat = np.zeros((n, m))
    for i in range(n):
        data = fp.readline().split()
        for j in range(m):
            lbs_mat[i, j] = float(data[j])
    fp.close()

np.save(open(output_weight_path, 'wb'), lbs_mat)