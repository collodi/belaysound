import matplotlib.pyplot as plt
import numpy as np

train = [0.985445, 0.972808]
test = [0.959649, 0.962934]

x = ['Clip Detection', 'Lane Detection']
x_ax = np.arange(len(x))

plt.bar(x_ax - 0.2, train, 0.38, label='Train')
plt.bar(x_ax + 0.2, test, 0.38, label='Test')
plt.xticks(x_ax, x)
plt.legend()

plt.ylim(0.9, 1)

plt.savefig('eval.pdf')
