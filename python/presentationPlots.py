import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

t = np.linspace(0, 4 * np.pi, 10000)

carrier = np.sin(4 * t)

# make a modulating wave, 1,1,0,0,1,1 and let 0 be low and 1 be high
bits = np.array([1, 0, 1, 1, 0])
modulation = np.repeat(bits, len(t) // len(bits))


modulated = carrier * (0.01 + 0.6 * modulation)  # 0.1 = base amplitude, 0.7 = switching depth

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t, modulated, color='pink', linewidth=4)
ax.axis('off')
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

plt.tight_layout(pad=0)
plt.savefig('modulated_signal.png', dpi=150, bbox_inches='tight',
            transparent=True, facecolor='none')
plt.show()      