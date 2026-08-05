import h5py
import numpy as np
import pandas as pd

f = h5py.File('q6_phase_a_dataset.h5', 'r')

# Get first 100 training samples
T_data = f['train/T'][:100]
y_data = f['train/y_30ch'][:100]
metadata = f['train/metadata'][:100]

# Create column names
T_cols = [f'T_{i}' for i in range(T_data.shape[1])]
y_cols = [f'y_{i}' for i in range(y_data.shape[1])]

# Combine into DataFrame
df = pd.DataFrame(np.hstack([T_data, y_data]), 
                  columns=T_cols + y_cols)

# Add metadata
df['base_profile_idx'] = metadata['base_profile_idx']
df['perturbation_type'] = [p.decode('utf-8') for p in metadata['perturbation_type']]
df['pc_idx'] = metadata['pc_idx']
df['sign'] = metadata['sign']
df['magnitude'] = metadata['magnitude']

# Save to CSV
df.to_csv('sample_data.csv', index=False)
print("✅ Saved first 100 samples to 'sample_data.csv'")

f.close()
