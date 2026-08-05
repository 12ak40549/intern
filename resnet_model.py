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

print("="*60)
print("RESIDUAL MLP VS PLAIN MLP COMPARISON")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📂 Loading data...")
f = h5py.File('q6_phase_a_dataset.h5', 'r')

X_train = f['train/T'][:]
y_train = f['train/y_30ch'][:]
X_test = f['test/T'][:]
y_test = f['test/y_30ch'][:]
X_val = f['val/T'][:]
y_val = f['val/y_30ch'][:]
f.close()

print(f"  Training: {X_train.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples")

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
# 5. DEFINE MODELS
# ============================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(x + self.net(x))

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64, output_dim=30, num_blocks=4):
        super(ResidualMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

class PlainMLP(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64, output_dim=30, num_layers=5):
        super(PlainMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ============================================
# 6. TRAINING FUNCTION
# ============================================
def train_model(model, train_loader, X_val_t, y_val_t, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_val_t = X_val_t.to(device)
    y_val_t = y_val_t.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
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
        
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
    
    return train_losses, val_losses, model

# ============================================
# 7. TRAIN BOTH MODELS
# ============================================
print("\n" + "="*60)
print("TRAINING RESIDUAL MLP (8 layers)")
print("="*60)
res_model = ResidualMLP(num_blocks=4)
print(f"  Parameters: {sum(p.numel() for p in res_model.parameters()):,}")
res_train_loss, res_val_loss, res_model_trained = train_model(
    res_model, train_loader, X_val_t, y_val_t, epochs=50
)

print("\n" + "="*60)
print("TRAINING PLAIN MLP (8 layers)")
print("="*60)
plain_model = PlainMLP(num_layers=8)
print(f"  Parameters: {sum(p.numel() for p in plain_model.parameters()):,}")
plain_train_loss, plain_val_loss, plain_model_trained = train_model(
    plain_model, train_loader, X_val_t, y_val_t, epochs=50
)

# ============================================
# 8. EVALUATE ON TEST SET
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, X_test_t, y_test_actual, scaler_y):
    model.eval()
    with torch.no_grad():
        X_test_t_device = X_test_t.to(device)
        y_pred_scaled = model(X_test_t_device).detach().cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2))
    return rmse, y_pred

res_rmse, res_pred = evaluate(res_model_trained, X_test_t, y_test, scaler_y)
plain_rmse, plain_pred = evaluate(plain_model_trained, X_test_t, y_test, scaler_y)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"✅ Residual MLP Test RMSE: {res_rmse:.4f}")
print(f"✅ Plain MLP Test RMSE: {plain_rmse:.4f}")
improvement = ((plain_rmse - res_rmse) / plain_rmse) * 100
print(f"📈 Improvement: {improvement:.2f}%")

# ============================================
# 9. PLOT RESULTS
# ============================================
print("\n📈 Generating plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss curves
axes[0,0].plot(res_train_loss, label='ResNet Train', color='green')
axes[0,0].plot(res_val_loss, label='ResNet Val', color='green', linestyle='--')
axes[0,0].plot(plain_train_loss, label='Plain Train', color='orange')
axes[0,0].plot(plain_val_loss, label='Plain Val', color='orange', linestyle='--')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Training Curves: ResNet vs Plain')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Predicted vs actual (ResNet)
for i in range(min(3, len(y_test))):
    axes[0,1].plot(y_test[i], 'b-', alpha=0.5, label='Actual' if i==0 else '')
    axes[0,2].plot(res_pred[i], 'g--', alpha=0.5, label='ResNet' if i==0 else '')

axes[0,1].set_title('Actual Test Samples')
axes[0,1].set_xlabel('Channel')
axes[0,1].set_ylabel('Value')
axes[0,1].grid(True, alpha=0.3)

axes[0,2].set_title('ResNet Predictions')
axes[0,2].set_xlabel('Channel')
axes[0,2].set_ylabel('Value')
axes[0,2].grid(True, alpha=0.3)

# Model comparison bar chart
models = ['ResNet', 'Plain MLP']
rmse_values = [res_rmse, plain_rmse]
colors = ['green', 'orange']
axes[1,0].bar(models, rmse_values, color=colors)
axes[1,0].set_ylabel('RMSE')
axes[1,0].set_title('Model Comparison')
axes[1,0].grid(True, alpha=0.3)
for i, v in enumerate(rmse_values):
    axes[1,0].text(i, v + 0.01, f'{v:.4f}', ha='center')

# Scatter plots
axes[1,1].scatter(y_test.flatten(), res_pred.flatten(), alpha=0.05, s=1, c='green')
axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1,1].set_xlabel('Actual')
axes[1,1].set_ylabel('Predicted')
axes[1,1].set_title(f'ResNet (RMSE: {res_rmse:.4f})')
axes[1,1].grid(True, alpha=0.3)

axes[1,2].scatter(y_test.flatten(), plain_pred.flatten(), alpha=0.05, s=1, c='orange')
axes[1,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1,2].set_xlabel('Actual')
axes[1,2].set_ylabel('Predicted')
axes[1,2].set_title(f'Plain MLP (RMSE: {plain_rmse:.4f})')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resnet_comparison.png', dpi=150)
print("✅ Saved plot to 'resnet_comparison.png'")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"CNN Test RMSE:        3.0970")
print(f"Residual MLP RMSE:    {res_rmse:.4f}")
print(f"Plain MLP RMSE:       {plain_rmse:.4f}")
if res_rmse < plain_rmse:
    print("✅ Residual MLP performed better than Plain MLP!")
else:
    print("⚠️ Plain MLP performed better in this case")
print("="*60)
