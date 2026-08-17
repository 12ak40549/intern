import os
os.environ['NUMPY_EXPERIMENTAL_DTYPE_API'] = '1'

import numpy as np
np._ARRAY_API = True
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("WEEK 3 - DAY 13: INVERSE PROBLEM BASELINE")
print("Signal (Measurements) → Profile (Temperature)")
print("="*70)

# ============================================
# 1. LOAD DATA (CORRECT INVERSE DIRECTION)
# ============================================
print("\n📂 Loading data...")
f = h5py.File('q6_phase_a_dataset.h5', 'r')

# INVERSE DIRECTION: Measurements → Profile
X_train = f['train/y_30ch'][:]     # Input: 30 measurements
y_train = f['train/T'][:]          # Output: 21 temperature levels

X_test = f['test/y_30ch'][:]       # Input: 30 measurements
y_test = f['test/T'][:]            # Output: 21 temperature levels

X_val = f['val/y_30ch'][:]         # Input: 30 measurements
y_val = f['val/T'][:]              # Output: 21 temperature levels

f.close()

print(f"  Training: {X_train.shape[0]} samples")
print(f"  Input (measurements): {X_train.shape[1]} channels")
print(f"  Output (temperature levels): {y_train.shape[1]} levels")
print(f"  ✅ CORRECT INVERSE DIRECTION: Signal → Profile")

# ============================================
# 2. STANDARDIZE DATA
# ============================================
print("\n📊 Standardizing data...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)
X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val)

# ============================================
# 3. CONVERT TO TENSORS
# ============================================
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled)
X_val_t = torch.FloatTensor(X_val_scaled)
y_val_t = torch.FloatTensor(y_val_scaled)

# ============================================
# 4. CREATE DATALOADER
# ============================================
batch_size = 64
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ============================================
# 5. DEFINE RESIDUAL MLP WITH DROPOUT
# ============================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(x + self.net(x))

class InverseResidualMLP(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=128, output_dim=21, num_blocks=4, dropout_rate=0.2):
        super(InverseResidualMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

model = InverseResidualMLP()
print(f"\n🧠 Model: Inverse Residual MLP")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 6. TRAINING WITH EARLY STOPPING & LR SCHEDULE
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"  Device: {device}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

print("\n🚀 Training with Early Stopping + Cosine LR Schedule...")

epochs = 200
train_losses = []
val_losses = []
learning_rates = []

# Early stopping
best_val_loss = float('inf')
patience_counter = 0
patience = 30
best_model_state = None

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        X_val_t_device = X_val_t.to(device)
        y_val_t_device = y_val_t.to(device)
        val_pred = model(X_val_t_device)
        val_loss = criterion(val_pred, y_val_t_device)
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss.item())
    learning_rates.append(scheduler.get_last_lr()[0])
    
    # Early stopping check
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
    
    # Update learning rate
    scheduler.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {val_loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    if patience_counter >= patience:
        print(f"  ⏹️ Early stopping at epoch {epoch+1}!")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"  ✅ Restored best model with val loss: {best_val_loss:.6f}")

# ============================================
# 7. EVALUATE ON TEST SET (In physical units)
# ============================================
print("\n📊 Evaluating on test set...")
model.eval()
with torch.no_grad():
    X_test_t_device = X_test_t.to(device)
    y_pred_scaled = model(X_test_t_device).detach().cpu().numpy()

# Inverse transform to physical units (Kelvin)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# RMSE per layer (in Kelvin)
layer_rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2, axis=0))
overall_rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2))

print(f"\n✅ Overall Test RMSE: {overall_rmse:.4f} K")
print(f"\n📊 Per-Layer RMSE (K):")
print(f"  First 5 levels: {layer_rmse[:5]}")
print(f"  Middle 5 levels: {layer_rmse[8:13]}")
print(f"  Last 5 levels: {layer_rmse[16:21]}")

# ============================================
# 8. PLOT RESULTS
# ============================================
print("\n📈 Generating plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss curves with early stopping
axes[0,0].plot(train_losses, label='Train Loss', color='blue')
axes[0,0].plot(val_losses, label='Val Loss', color='orange')
if len(val_losses) < epochs:
    axes[0,0].axvline(x=len(val_losses)-1, color='r', linestyle='--', label='Early Stop')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Training Curves with Early Stopping')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Learning rate schedule
axes[0,1].plot(learning_rates, color='green')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Learning Rate')
axes[0,1].set_title('Cosine Annealing LR Schedule')
axes[0,1].grid(True, alpha=0.3)

# Predicted vs actual (first 3 test samples)
for i in range(min(3, len(y_test_actual))):
    axes[0,2].plot(y_test_actual[i], 'b-', alpha=0.5, label='Actual' if i==0 else '')
    axes[0,2].plot(y_pred[i], 'r--', alpha=0.5, label='Predicted' if i==0 else '')
axes[0,2].set_xlabel('Level (21 levels)')
axes[0,2].set_ylabel('Temperature (K)')
axes[0,2].set_title('Predicted vs Actual (Test Set)')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# Per-layer RMSE
axes[1,0].bar(range(len(layer_rmse)), layer_rmse, color='purple')
axes[1,0].set_xlabel('Level')
axes[1,0].set_ylabel('RMSE (K)')
axes[1,0].set_title(f'RMSE per Atmospheric Level (Overall: {overall_rmse:.3f} K)')
axes[1,0].grid(True, alpha=0.3)

# Scatter plot
axes[1,1].scatter(y_test_actual.flatten(), y_pred.flatten(), alpha=0.05, s=1)
axes[1,1].plot([y_test_actual.min(), y_test_actual.max()], 
               [y_test_actual.min(), y_test_actual.max()], 'r--', linewidth=2)
axes[1,1].set_xlabel('Actual Temperature (K)')
axes[1,1].set_ylabel('Predicted Temperature (K)')
axes[1,1].set_title(f'Prediction Scatter (RMSE: {overall_rmse:.3f} K)')
axes[1,1].grid(True, alpha=0.3)

# Error distribution
errors = (y_pred - y_test_actual).flatten()
axes[1,2].hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[1,2].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1,2].set_xlabel('Prediction Error (K)')
axes[1,2].set_ylabel('Frequency')
axes[1,2].set_title(f'Error Distribution (Std: {np.std(errors):.3f} K)')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day13_inverse_baseline.png', dpi=150)
print("✅ Saved plot to 'day13_inverse_baseline.png'")

# ============================================
# 9. SUMMARY
# ============================================
print("\n" + "="*70)
print("DAY 13 SUMMARY - INVERSE BASELINE")
print("="*70)
print(f"✅ Overall Test RMSE: {overall_rmse:.4f} K")
print(f"✅ Best Validation Loss: {best_val_loss:.6f}")
print(f"✅ Early Stopping at: {len(val_losses)} epochs")
print(f"\n📊 Per-Layer RMSE (K):")
for i, rmse in enumerate(layer_rmse[:10]):
    print(f"  Level {i+1:2d}: {rmse:.4f} K")
print(f"  ...")
for i, rmse in enumerate(layer_rmse[15:21]):
    print(f"  Level {i+16:2d}: {rmse:.4f} K")

print("\n" + "="*70)
print("✅ Day 13 Complete! Inverse baseline established.")
print("="*70)
