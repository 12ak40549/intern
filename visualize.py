import h5py
import matplotlib.pyplot as plt
import numpy as np

f = h5py.File('q6_phase_a_dataset.h5', 'r')

# Get first sample
sample_T = f['train/T'][0]
sample_y = f['train/y_30ch'][0]

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot temperature profile
ax1.plot(sample_T, 'b-o', markersize=4)
ax1.set_title('Temperature Profile (T) - Sample 0')
ax1.set_xlabel('Vertical Level (0-20)')
ax1.set_ylabel('Temperature (K)')
ax1.grid(True, alpha=0.3)

# Plot output channels
ax2.plot(sample_y, 'r-o', markersize=4)
ax2.set_title('Output Channels (y_30ch) - Sample 0')
ax2.set_xlabel('Channel Index (0-29)')
ax2.set_ylabel('Value')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

f.close()
