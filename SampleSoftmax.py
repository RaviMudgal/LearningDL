"""Softmax."""
import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e = np.exp(x)
    dist = e/np.sum(e,axis=0)
    return dist
      # TODO: Compute and return softmax(x)

scores = [3.0, 1.0, 0.2]
print(softmax(scores))

# Plot softmax curves

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
