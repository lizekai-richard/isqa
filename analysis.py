import json
import numpy as np
import matplotlib.pyplot as plt


x = [0, 200, 400, 600, 800, 1000, 1200]
rouge1 = [10, 11, 12, 13, 14, 15, 16]
rouge2 = [1, 2, 3, 4, 5, 6, 7]
rougeL = [20, 21, 22, 23, 24, 25, 26]

plt.style.use("ggplot")
plt.plot(x, rouge1, 'r--', label='rouge1')
plt.plot(x, rouge2, 'g--', label='rouge2')
plt.plot(x, rougeL, 'b--', label='rougeL')
plt.plot(x, rouge1, 'ro-', x, rouge2, 'g+-', x, rougeL, 'b^-')
plt.title('The Lasers in Three Conditions')
plt.xlabel('eval_step')
plt.ylabel('%')
plt.legend()
plt.grid(True)
plt.show()
