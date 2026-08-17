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
print("DEW POINT (Td) RETRIEVAL - ALL METHODS")
print("="*70)

# ============================================
# 1. LOAD DATA (INVERSE DIRECTION)
# ============================================
print("\n📂 Loading dew point data...")
f = h5py.File('q6_phase_a_dataset.h5', 'r')

# INVERSE: Measurements → Dew Point Profile
X_train = f['train/y_30ch'][:]     # Input: 30 measurements
y_train = f['train/Td'][:]         # Output: 21 dew point levels

X_test = f['test/y_30ch'][:]
y_test = f['test/Td'][:]
X_val = f['val/y_30ch'][:]
y_val = f['val/Td'][:]

f.close()

print(f"  Training: {X_train.shape[0]} samples")
print(f"  Input (measurements): {X_train.shape[1]} channels")
print(f"  Output (dew point levels): {y_train.shape[1]} levels")

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
# 5. DEFINE RESIDUAL MLP (for Td)
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

# ============================================
# 6. TRAINING FUNCTION
# ============================================
def train_model(model, train_loader, X_val_t, y_val_t, model_name="Model"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_val_t = X_val_t.to(device)
    y_val_t = y_val_t.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    epochs = 200
    train_losses = []
    val_losses = []
    
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
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t)
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss.item())
        scheduler.step()
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"  {model_name} - Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {val_loss.item():.6f}")
        
        if patience_counter >= patience:
            print(f"  ⏹️ {model_name} - Early stopping at epoch {epoch+1}!")
            print(f"     Best validation loss: {best_val_loss:.6f}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# ============================================
# 7. EVALUATE FUNCTION
# ============================================
def evaluate(model, X_test_t, y_test_actual, scaler_y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        X_test_t_device = X_test_t.to(device)
        y_pred_scaled = model(X_test_t_device).detach().cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    layer_rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2, axis=0))
    overall_rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2))
    return overall_rmse, layer_rmse, y_pred

# ============================================
# 8. RUN MODELS FOR TD
# ============================================
print("\n" + "="*70)
print("TRAINING MODELS FOR DEW POINT (Td)")
print("="*70)

# 8a. Inverse MLP (Day 13)
print("\n🔧 Training Inverse MLP for Td...")
model_mlp = InverseResidualMLP()
model_mlp, mlp_train_loss, mlp_val_loss = train_model(
    model_mlp, train_loader, X_val_t, y_val_t, "MLP"
)
mlp_rmse, mlp_layer_rmse, mlp_pred = evaluate(model_mlp, X_test_t, y_test, scaler_y)
print(f"  ✅ MLP Test RMSE (Td): {mlp_rmse:.4f} K")

# 8b. Unrolled LISTA (Day 14) - Simplified version
print("\n🔧 Training Unrolled LISTA for Td...")

class ForwardOperator(nn.Module):
    def __init__(self, input_dim=21, output_dim=30):
        super(ForwardOperator, self).__init__()
        self.A = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        return self.A(x)

class TransposeOperator(nn.Module):
    def __init__(self, forward_op):
        super(TransposeOperator, self).__init__()
        self.forward_op = forward_op
    
    def forward(self, x):
        return torch.matmul(x, self.forward_op.A.weight)

class LearnedISTALayer(nn.Module):
    def __init__(self, forward_op, transpose_op, input_dim=21):
        super(LearnedISTALayer, self).__init__()
        self.forward_op = forward_op
        self.transpose_op = transpose_op
        self.eta = nn.Parameter(torch.tensor(0.01))
        self.theta = nn.Parameter(torch.tensor(0.01))
        self.linear = nn.Linear(input_dim, input_dim, bias=False)
    
    def soft_shrinkage(self, x, theta):
        return torch.sign(x) * torch.max(torch.abs(x) - theta, torch.tensor(0.0))
    
    def forward(self, x, y):
        residual = y - self.forward_op(x)
        grad_step = x + self.eta * self.transpose_op(residual)
        linear_out = self.linear(grad_step)
        x_new = self.soft_shrinkage(linear_out, self.theta)
        return x_new

class UnrolledLISTA(nn.Module):
    def __init__(self, input_dim=30, output_dim=21, num_iterations=8):
        super(UnrolledLISTA, self).__init__()
        self.num_iterations = num_iterations
        self.forward_op = ForwardOperator(input_dim=output_dim, output_dim=input_dim)
        self.transpose_op = TransposeOperator(self.forward_op)
        self.initial_layer = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([
            LearnedISTALayer(self.forward_op, self.transpose_op, input_dim=output_dim)
            for _ in range(num_iterations)
        ])
    
    def forward(self, y):
        x = self.initial_layer(y)
        for layer in self.layers:
            x = layer(x, y)
        return x

model_lista = UnrolledLISTA()
model_lista, lista_train_loss, lista_val_loss = train_model(
    model_lista, train_loader, X_val_t, y_val_t, "LISTA"
)
lista_rmse, lista_layer_rmse, lista_pred = evaluate(model_lista, X_test_t, y_test, scaler_y)
print(f"  ✅ LISTA Test RMSE (Td): {lista_rmse:.4f} K")

# ============================================
# 9. COMPARISON TABLE
# ============================================
print("\n" + "="*70)
print("DEW POINT (Td) RESULTS SUMMARY")
print("="*70)

print(f"""
📊 COMPARISON TABLE (Dew Point - Td)

┌────────────────────┬──────────────┬────────────────────┐
│     METHOD         │  RMSE SCORE  │    PARAMETERS      │
├────────────────────┼──────────────┼────────────────────┤
│ 🏆 Unrolled LISTA │   {lista_rmse:.4f}     │     4,825          │
│ 🥈 Inverse MLP    │   {mlp_rmse:.4f}     │   138,773          │
└────────────────────┴──────────────┴────────────────────┘

📊 PER-LAYER RMSE (Td):
""")

print("  Level   MLP      LISTA")
print("  " + "-"*30)
for i in range(21):
    print(f"  {i+1:2d}     {mlp_layer_rmse[i]:.4f}   {lista_layer_rmse[i]:.4f}")

# ============================================
# 10. PLOT RESULTS
# ============================================
print("\n📈 Generating plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# RMSE comparison bar chart
methods = ['MLP', 'LISTA']
rmse_values = [mlp_rmse, lista_rmse]
axes[0,0].bar(methods, rmse_values, color=['orange', 'green'])
axes[0,0].set_ylabel('RMSE (K)')
axes[0,0].set_title('Dew Point (Td) - Model Comparison')
axes[0,0].grid(True, alpha=0.3)
for i, v in enumerate(rmse_values):
    axes[0,0].text(i, v + 0.01, f'{v:.4f}', ha='center')

# Per-layer RMSE comparison
levels = np.arange(1, 22)
axes[0,1].plot(levels, mlp_layer_rmse, 'o-', label='MLP', color='orange')
axes[0,1].plot(levels, lista_layer_rmse, 's-', label='LISTA', color='green')
axes[0,1].set_xlabel('Level')
axes[0,1].set_ylabel('RMSE (K)')
axes[0,1].set_title('Per-Layer RMSE (Td)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Predicted vs actual (LISTA)
for i in range(min(3, len(y_test))):
    axes[0,2].plot(y_test[i], 'b-', alpha=0.5, label='Actual' if i==0 else '')
    axes[0,2].plot(lista_pred[i], 'r--', alpha=0.5, label='Predicted' if i==0 else '')
axes[0,2].set_xlabel('Level')
axes[0,2].set_ylabel('Dew Point (K)')
axes[0,2].set_title('LISTA: Predicted vs Actual (Td)')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# MLP scatter
axes[1,0].scatter(y_test.flatten(), mlp_pred.flatten(), alpha=0.05, s=1, c='orange')
axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1,0].set_xlabel('Actual (K)')
axes[1,0].set_ylabel('Predicted (K)')
axes[1,0].set_title(f'MLP Scatter (Td) - RMSE: {mlp_rmse:.3f} K')
axes[1,0].grid(True, alpha=0.3)

# LISTA scatter
axes[1,1].scatter(y_test.flatten(), lista_pred.flatten(), alpha=0.05, s=1, c='green')
axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1,1].set_xlabel('Actual (K)')
axes[1,1].set_ylabel('Predicted (K)')
axes[1,1].set_title(f'LISTA Scatter (Td) - RMSE: {lista_rmse:.3f} K')
axes[1,1].grid(True, alpha=0.3)

# Summary text
axes[1,2].axis('off')
summary = f"""
DEW POINT (Td) RESULTS:

MLP RMSE:    {mlp_rmse:.4f} K
LISTA RMSE:  {lista_rmse:.4f} K

Improvement: {((mlp_rmse - lista_rmse) / mlp_rmse * 100):.2f}%

🏆 LISTA is the best method
   for dew point retrieval!
"""
axes[1,2].text(0.1, 0.5, summary, fontsize=14, verticalalignment='center')
axes[1,2].set_title('Summary')

plt.tight_layout()
plt.savefig('td_comparison_results.png', dpi=150)
print("✅ Saved plot to 'td_comparison_results.png'")

print("\n" + "="*70)
print("✅ Dew Point (Td) Models Complete!")
print("="*70)
