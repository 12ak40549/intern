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
import warnings
warnings.filterwarnings('ignore')

# Try to import deepinv
try:
    import deepinv as dinv
    DEEPINV_AVAILABLE = True
    print("✅ deepinv imported successfully!")
except ImportError:
    DEEPINV_AVAILABLE = False
    print("⚠️ deepinv not installed. Using built-in denoiser instead.")
    print("   To install: pip install deepinv")

print("="*70)
print("WEEK 3 - DAY 15: PLUG-AND-PLAY (PnP-ADMM)")
print("Modular inverse problem solving with learned denoiser")
print("="*70)

# ============================================
# 1. LOAD DATA (INVERSE DIRECTION)
# ============================================
print("\n📂 Loading data...")
f = h5py.File('q6_phase_a_dataset.h5', 'r')

X_train = f['train/y_30ch'][:]
y_train = f['train/T'][:]
X_test = f['test/y_30ch'][:]
y_test = f['test/T'][:]
X_val = f['val/y_30ch'][:]
y_val = f['val/T'][:]

f.close()

print(f"  Training: {X_train.shape[0]} samples")
print(f"  Input (measurements): {X_train.shape[1]} channels")
print(f"  Output (temperature levels): {y_train.shape[1]} levels")

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
# 5. DEFINE FORWARD OPERATOR A
# ============================================
class ForwardOperator(nn.Module):
    """Forward physics: profile → measurements"""
    def __init__(self, input_dim=21, output_dim=30):
        super(ForwardOperator, self).__init__()
        self.A = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        return self.A(x)

# ============================================
# 6. DEFINE DENOISER (DnCNN-style)
# ============================================
class Denoiser(nn.Module):
    """
    Simple 1D denoiser for atmospheric profiles
    """
    def __init__(self, input_dim=21, hidden_dim=128):
        super(Denoiser, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        # Residual learning: learn the noise
        noise = self.net(x)
        return x - noise  # Denoised output

# ============================================
# 7. TRAIN THE DENOISER
# ============================================
print("\n🔧 Training Denoiser...")

denoiser = Denoiser()
denoiser_optimizer = optim.Adam(denoiser.parameters(), lr=0.001)
denoiser_criterion = nn.MSELoss()

# Create noisy training data
noise_level = 0.1
X_train_clean = y_train_t[:10000]  # Use subset for speed
X_train_noisy = X_train_clean + noise_level * torch.randn_like(X_train_clean)

denoiser_epochs = 20
denoiser_losses = []

for epoch in range(denoiser_epochs):
    denoiser.train()
    epoch_loss = 0
    batch_size_d = 128
    num_batches = 0
    for i in range(0, len(X_train_noisy), batch_size_d):
        batch_clean = X_train_clean[i:i+batch_size_d]
        batch_noisy = X_train_noisy[i:i+batch_size_d]
        
        denoiser_optimizer.zero_grad()
        denoised = denoiser(batch_noisy)
        loss = denoiser_criterion(denoised, batch_clean)
        loss.backward()
        denoiser_optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / max(num_batches, 1)
    denoiser_losses.append(avg_loss)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{denoiser_epochs} - Denoiser Loss: {avg_loss:.6f}")

print(f"  ✅ Denoiser trained!")

# ============================================
# 8. PNP-ADMM SOLVER (FIXED)
# ============================================
def pnp_admm(y, forward_op, denoiser, num_iterations=20, rho=0.1, lr=0.01):
    """
    Plug-and-Play ADMM solver
    
    Solves: min_x ||y - A(x)||² + λ·D(x)
    using ADMM with denoiser D
    """
    device = next(forward_op.parameters()).device
    batch_size = y.shape[0]
    x_dim = 21
    
    # Initialize
    x = torch.randn(batch_size, x_dim, device=device) * 0.01
    z = torch.zeros_like(x)
    u = torch.zeros_like(x)
    
    history = []
    
    for k in range(num_iterations):
        # 1. Data consistency step (x-update) with gradient descent
        x = x.clone().detach().requires_grad_(True)
        
        # Compute loss
        data_term = torch.mean((y - forward_op(x))**2)
        prox_term = torch.mean((x - z + u)**2)
        loss = data_term + (rho/2) * prox_term
        
        # Gradient step
        loss.backward()
        with torch.no_grad():
            x = x - lr * x.grad
            x = x.detach()
        
        # 2. Denoising step (z-update)
        with torch.no_grad():
            z = denoiser(x + u)
        
        # 3. Dual variable update
        u = u + (x - z)
        
        history.append(loss.item())
    
    return x, history

# ============================================
# 9. INSTANTIATE FORWARD OPERATOR
# ============================================
forward_op = ForwardOperator()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
forward_op = forward_op.to(device)
denoiser = denoiser.to(device)

print(f"\n🧠 PnP-ADMM Configuration:")
print(f"  Device: {device}")
print(f"  Iterations: 20")
print(f"  Denoiser parameters: {sum(p.numel() for p in denoiser.parameters()):,}")

# ============================================
# 10. EVALUATE ON TEST SET
# ============================================
print("\n📊 Evaluating PnP-ADMM on test set...")

y_test_t_device = X_test_t.to(device)  # Measurements
y_test_actual = y_test  # True profiles

# Process in batches to avoid memory issues
batch_size_pnp = 32
all_predictions = []
all_losses = []

for i in range(0, len(y_test_t_device), batch_size_pnp):
    batch_y = y_test_t_device[i:i+batch_size_pnp]
    
    # Run PnP-ADMM
    x_pred, history = pnp_admm(
        batch_y, 
        forward_op, 
        denoiser,
        num_iterations=15,
        rho=0.1,
        lr=0.01
    )
    
    all_predictions.append(x_pred.cpu())
    all_losses.append(history)

y_pred = torch.cat(all_predictions, dim=0).numpy()

# Inverse transform to physical units
y_pred = scaler_y.inverse_transform(y_pred)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# Calculate RMSE
layer_rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2, axis=0))
overall_rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2))

print(f"\n✅ Overall Test RMSE: {overall_rmse:.4f} K")

# ============================================
# 11. PLOT RESULTS
# ============================================
print("\n📈 Generating plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# ADMM convergence (first batch)
axes[0,0].plot(all_losses[0] if all_losses else [])
axes[0,0].set_xlabel('ADMM Iteration')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('PnP-ADMM Convergence')
axes[0,0].grid(True, alpha=0.3)

# Denoiser training loss
axes[0,1].plot(denoiser_losses)
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss')
axes[0,1].set_title('Denoiser Training')
axes[0,1].grid(True, alpha=0.3)

# Predicted vs actual
for i in range(min(3, len(y_test_actual))):
    axes[0,2].plot(y_test_actual[i], 'b-', alpha=0.5, label='Actual' if i==0 else '')
    axes[0,2].plot(y_pred[i], 'r--', alpha=0.5, label='Predicted' if i==0 else '')
axes[0,2].set_xlabel('Level')
axes[0,2].set_ylabel('Temperature (K)')
axes[0,2].set_title('PnP-ADMM: Predicted vs Actual')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# Per-layer RMSE
axes[1,0].bar(range(len(layer_rmse)), layer_rmse, color='green')
axes[1,0].set_xlabel('Level')
axes[1,0].set_ylabel('RMSE (K)')
axes[1,0].set_title(f'PnP-ADMM RMSE per Level (Overall: {overall_rmse:.3f} K)')
axes[1,0].grid(True, alpha=0.3)

# Scatter plot
axes[1,1].scatter(y_test_actual.flatten(), y_pred.flatten(), alpha=0.05, s=1, c='green')
axes[1,1].plot([y_test_actual.min(), y_test_actual.max()], 
               [y_test_actual.min(), y_test_actual.max()], 'r--', linewidth=2)
axes[1,1].set_xlabel('Actual (K)')
axes[1,1].set_ylabel('Predicted (K)')
axes[1,1].set_title(f'PnP-ADMM Scatter (RMSE: {overall_rmse:.3f} K)')
axes[1,1].grid(True, alpha=0.3)

# Error distribution
errors = (y_pred - y_test_actual).flatten()
axes[1,2].hist(errors, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1,2].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1,2].set_xlabel('Error (K)')
axes[1,2].set_ylabel('Frequency')
axes[1,2].set_title(f'Error Distribution (Std: {np.std(errors):.3f} K)')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day15_pnp_admm.png', dpi=150)
print("✅ Saved plot to 'day15_pnp_admm.png'")

# ============================================
# 12. COMPARE ALL METHODS
# ============================================
print("\n" + "="*70)
print("DAY 15 SUMMARY - PLUG-AND-PLAY ADMM")
print("="*70)
print(f"✅ Overall Test RMSE: {overall_rmse:.4f} K")
print(f"✅ Denoiser parameters: {sum(p.numel() for p in denoiser.parameters()):,}")

# Compare with previous days
day13_rmse = 1.1090
day14_rmse = 0.8834
print(f"\n📊 Comparison of All Methods:")
print(f"  Day 13 (Standard MLP):      {day13_rmse:.4f} K")
print(f"  Day 14 (Unrolled LISTA):    {day14_rmse:.4f} K")
print(f"  Day 15 (PnP-ADMM):          {overall_rmse:.4f} K")

best_rmse = min(day13_rmse, day14_rmse, overall_rmse)
if overall_rmse == best_rmse:
    print(f"\n🏆 PnP-ADMM is the BEST method so far!")
elif day14_rmse == best_rmse:
    print(f"\n🏆 Unrolled LISTA is still the BEST method so far!")
else:
    print(f"\n🏆 Day 13 MLP is still the BEST method so far!")

print("\n" + "="*70)
print("✅ Day 15 Complete! PnP-ADMM tested.")
print("="*70)
