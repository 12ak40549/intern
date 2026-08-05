import h5py
import numpy as np

f = h5py.File('q6_phase_a_dataset.h5', 'r')

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

# Training data stats
train_T = f['train/T'][:]
train_y = f['train/y_30ch'][:]

print(f"\n📊 TRAINING SET:")
print(f"  Samples: {train_T.shape[0]}")
print(f"  Input features (T): {train_T.shape[1]}")
print(f"  Output channels (y): {train_y.shape[1]}")
print(f"  T - min: {train_T.min():.2f}, max: {train_T.max():.2f}, mean: {train_T.mean():.2f}")
print(f"  y - min: {train_y.min():.2f}, max: {train_y.max():.2f}, mean: {train_y.mean():.2f}")

# Test data
test_T = f['test/T'][:]
test_y = f['test/y_30ch'][:]
print(f"\n📊 TEST SET:")
print(f"  Samples: {test_T.shape[0]}")
print(f"  T - min: {test_T.min():.2f}, max: {test_T.max():.2f}, mean: {test_T.mean():.2f}")

# Validation data
val_T = f['val/T'][:]
val_y = f['val/y_30ch'][:]
print(f"\n📊 VALIDATION SET:")
print(f"  Samples: {val_T.shape[0]}")

# Metadata stats
metadata = f['train/metadata'][:]
unique_base = np.unique(metadata['base_profile_idx'])
unique_perturb = np.unique(metadata['perturbation_type'])
print(f"\n📊 METADATA:")
print(f"  Unique base profiles: {len(unique_base)}")
print(f"  Perturbation types: {unique_perturb}")
print(f"  Magnitudes: {np.unique(metadata['magnitude'])}")

f.close()
print("\n✅ Done!")
