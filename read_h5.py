import h5py
import numpy as np

# ========== STEP 4A: OPEN THE FILE ==========
# Replace 'q6_phase_a_dataset.h55' with your actual file name
file_path = 'q6_phase_a_dataset.h5'  

# 'r' means read-only mode (safe, won't modify the file)
file = h5py.File(file_path, 'r')

print("=" * 50)
print("FILE OPENED SUCCESSFULLY!")
print("=" * 50)

# ========== STEP 4B: EXPLORE THE STRUCTURE ==========
# See what's inside - top-level groups and datasets
print("\nTop-level contents:")
print(file.keys())  # This shows all groups/datasets at the root

print("\nDetailed structure:")
def print_structure(name, obj):
    """Recursively print everything in the file"""
    indent = "  " * name.count('/')  # Indent based on depth
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}📊 {name} (Dataset - Shape: {obj.shape}, Type: {obj.dtype})")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}📁 {name} (Group)")

# This will recursively explore your file
file.visititems(print_structure)

# ========== STEP 4C: READ SPECIFIC DATA ==========
# Let's say you found a dataset named 'data' or 'images' or 'values'
# You need to replace 'dataset_name' with the actual name you saw above

dataset_name = 'dataset_name'  # CHANGE THIS!

if dataset_name in file:
    print(f"\nReading dataset: {dataset_name}")
    data = file[dataset_name][:]  # The [:] reads all the data
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"\nFirst few rows/values:\n{data[:5]}")  # Show first 5 entries
    
    # If it's an image, you can check its shape
    if len(data.shape) >= 2:
        print(f"Looks like a 2D array (like an image) of size {data.shape}")
else:
    print(f"\n⚠️ Dataset '{dataset_name}' not found. Check the keys above.")

# ========== STEP 4D: CLOSE THE FILE ==========
file.close()
print("\n✅ File closed successfully!")
