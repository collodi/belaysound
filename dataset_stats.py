import numpy as np
import sys

fn = sys.argv[1]

with open(fn) as f:
	nums = [float(ln) for ln in f]

	avg = np.mean(nums)
	mn = np.min(nums)
	mx = np.max(nums)

	print(f'mean: {avg}')
	print(f'min: {mn}')
	print(f'max: {mx}')
