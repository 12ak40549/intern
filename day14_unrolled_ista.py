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
print("WEEK 3 - DAY 14: ALGORITHM UNROLLING (LISTA)")
print("Learned ISTA with physics-based layers")
print("="*70)

# ============================================
# 1. LOAD DATA (INVERSE DIRECTION)
# ============================================
print("\n📂 Loading data...")
f = h5py.File('q6_phase_a_dataset.h5', 'r')

# INVERSE: Measurements → Profile
X_train = f['train/y_30ch'][:]     # Input: 30 measurements
y_train = f['train/T'][:]          # Output: 21 temperature levels
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
    """Learned approximation of the forward physics A: profile → measurements"""
    def __init__(self, input_dim=21, output_dim=30):
        super(ForwardOperator, self).__init__()
        # A: (output_dim, input_dim) = (30, 21)
        self.A = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        # x is (batch, 21) → output is (batch, 30)
        return self.A(x)

class TransposeOperator(nn.Module):
    """Transpose of forward operator: measurements → profile"""
    def __init__(self, forward_operator):
        super(TransposeOperator, self).__init__()
        self.forward_op = forward_operator
    
    def forward(self, x):
        # A is (output_dim, input_dim) = (30, 21)
        # A.weight is (30, 21)
        # x is (batch, 30)
        # x @ A.weight → (batch, 21) ✓
        return torch.matmul(x, self.forward_op.A.weight)

# ============================================
# 6. DEFINE LEARNED ISTA LAYER
# ============================================
class LearnedISTALayer(nn.Module):
    """
    One unrolled ISTA iteration:
    x_{k+1} = shrink(x_k + eta * A^T(y - A*x_k), theta)
    """
    def __init__(self, forward_op, transpose_op, input_dim=21):
        super(LearnedISTALayer, self).__init__()
        self.forward_op = forward_op
        self.transpose_op = transpose_op
        # Learnable step size (eta)
        self.eta = nn.Parameter(torch.tensor(0.01))
        # Learnable threshold (theta)
        self.theta = nn.Parameter(torch.tensor(0.01))
        # Learnable linear layer after shrinkage (optional)
        self.linear = nn.Linear(input_dim, input_dim, bias=False)
    
    def soft_shrinkage(self, x, theta):
        """Soft thresholding: sign(x) * max(|x| - theta, 0)"""
        return torch.sign(x) * torch.max(torch.abs(x) - theta, torch.tensor(0.0))
    
    def forward(self, x, y):
        # Residual: y - A*x
        residual = y - self.forward_op(x)
        # Gradient step: x + eta * A^T * residual
        grad_step = x + self.eta * self.transpose_op(residual)
        # Linear transformation
        linear_out = self.linear(grad_step)
        # Shrinkage (soft thresholding)
        x_new = self.soft_shrinkage(linear_out, self.theta)
        return x_new

# ============================================
# 7. DEFINE UNROLLED LISTA NETWORK
# ============================================
class UnrolledLISTA(nn.Module):
    """
    K iterations of learned ISTA unrolled as a network
    """
    def __init__(self, input_dim=30, output_dim=21, num_iterations=8):
        super(UnrolledLISTA, self).__init__()
        self.num_iterations = num_iterations
        self.output_dim = output_dim
        
        # Initialize forward operator
        self.forward_op = ForwardOperator(input_dim=output_dim, output_dim=input_dim)
        self.transpose_op = TransposeOperator(self.forward_op)
        
        # Initialize layers
        self.initial_layer = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([
            LearnedISTALayer(self.forward_op, self.transpose_op, input_dim=output_dim)
            for _ in range(num_iterations)
        ])
    
    def forward(self, y):
        # Initial guess: linear projection of measurements
        x = self.initial_layer(y)
        
        # Unrolled iterations
        for layer in self.layers:
            x = layer(x, y)
        
        return x

# ============================================
# 8. INSTANTIATE MODEL
# ============================================
model = UnrolledLISTA(num_iterations=8)
print(f"\n🧠 Model: Unrolled LISTA ({model.num_iterations} iterations)")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"  Device: {device}")

# ============================================
# 9. TRAINING WITH EARLY STOPPING
# ============================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

print("\n🚀 Training Unrolled LISTA...")
print(f"  This learns step sizes (eta) and thresholds (theta) for each iteration")

epochs = 100
train_losses = []
val_losses = []
learning_rates = []

# Early stopping
best_val_loss = float('inf')
patience_counter = 0
patience = 20
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
    
    # Early stopping
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
    
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {val_loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    if patience_counter >= patience:
        print(f"  ⏹️ Early stopping at epoch {epoch+1}!")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"  ✅ Restored best model with val loss: {best_val_loss:.6f}")

# ============================================
# 10. EVALUATE ON TEST SET
# ============================================
print("\n📊 Evaluating on test set...")
model.eval()
with torch.no_grad():
    X_test_t_device = X_test_t.to(device)
    y_pred_scaled = model(X_test_t_device).detach().cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

layer_rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2, axis=0))
overall_rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2))

print(f"\n✅ Overall Test RMSE: {overall_rmse:.4f} K")

# ============================================
# 11. PRINT LEARNED PARAMETERS
# ============================================
print(f"\n📊 Learned Parameters:")
for i, layer in enumerate(model.layers):
    print(f"  Iteration {i+1}: eta = {layer.eta.item():.4f}, theta = {layer.theta.item():.4f}")

# ============================================
# 12. PLOT RESULTS
# ============================================
print("\n📈 Generating plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss curves
axes[0,0].plot(train_losses, label='Train Loss', color='blue')
axes[0,0].plot(val_losses, label='Val Loss', color='orange')
if len(val_losses) < epochs:
    axes[0,0].axvline(x=len(val_losses)-1, color='r', linestyle='--', label='Early Stop')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('LISTA Training Curves')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Learning rate
axes[0,1].plot(learning_rates, color='green')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Learning Rate')
axes[0,1].set_title('Cosine Annealing LR Schedule')
axes[0,1].grid(True, alpha=0.3)

# Predicted vs actual
for i in range(min(3, len(y_test_actual))):
    axes[0,2].plot(y_test_actual[i], 'b-', alpha=0.5, label='Actual' if i==0 else '')
    axes[0,2].plot(y_pred[i], 'r--', alpha=0.5, label='Predicted' if i==0 else '')
axes[0,2].set_xlabel('Level')
axes[0,2].set_ylabel('Temperature (K)')
axes[0,2].set_title('LISTA: Predicted vs Actual')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# Per-layer RMSE
axes[1,0].bar(range(len(layer_rmse)), layer_rmse, color='purple')
axes[1,0].set_xlabel('Level')
axes[1,0].set_ylabel('RMSE (K)')
axes[1,0].set_title(f'LISTA RMSE per Level (Overall: {overall_rmse:.3f} K)')
axes[1,0].grid(True, alpha=0.3)

# Scatter plot
axes[1,1].scatter(y_test_actual.flatten(), y_pred.flatten(), alpha=0.05, s=1)
axes[1,1].plot([y_test_actual.min(), y_test_actual.max()], 
               [y_test_actual.min(), y_test_actual.max()], 'r--', linewidth=2)
axes[1,1].set_xlabel('Actual (K)')
axes[1,1].set_ylabel('Predicted (K)')
axes[1,1].set_title(f'LISTA Scatter (RMSE: {overall_rmse:.3f} K)')
axes[1,1].grid(True, alpha=0.3)

# Error distribution
errors = (y_pred - y_test_actual).flatten()
axes[1,2].hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[1,2].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1,2].set_xlabel('Error (K)')
axes[1,2].set_ylabel('Frequency')
axes[1,2].set_title(f'LISTA Error Distribution (Std: {np.std(errors):.3f} K)')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day14_unrolled_ista.png', dpi=150)
print("✅ Saved plot to 'day14_unrolled_ista.png'")

# ============================================
# 13. COMPARE WITH DAY 13
# ============================================
print("\n" + "="*70)
print("DAY 14 SUMMARY - UNROLLED LISTA")
print("="*70)
print(f"✅ Overall Test RMSE: {overall_rmse:.4f} K")
print(f"✅ Learned {len(model.layers)} ISTA iterations")
print(f"✅ Early stopping at: {len(val_losses)} epochs")

# Compare with Day 13 baseline
day13_rmse = 1.1090
improvement = ((day13_rmse - overall_rmse) / day13_rmse) * 100
print(f"\n📊 Comparison with Day 13 Baseline:")
print(f"  Day 13 (Standard MLP): {day13_rmse:.4f} K")
print(f"  Day 14 (Unrolled LISTA): {overall_rmse:.4f} K")
print(f"  Improvement: {improvement:.2f}%")

print("\n" + "="*70)
print("✅ Day 14 Complete! Unrolled LISTA trained.")
print("="*70)
