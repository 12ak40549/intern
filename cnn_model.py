import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ============================================
# 1. LOAD DATA
# ============================================
print("Loading data...")
f = h5py.File('q6_phase_a_dataset.h5', 'r')

# Load training data
X_train = f['train/T'][:]  # (150025, 21)
y_train = f['train/y_30ch'][:]  # (150025, 30)

# Load test data
X_test = f['test/T'][:]  # (221, 21)
y_test = f['test/y_30ch'][:]  # (221, 30)

# Load validation data
X_val = f['val/T'][:]  # (221, 21)
y_val = f['val/y_30ch'][:]  # (221, 30)

f.close()

print(f"Training: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")

# ============================================
# 2. STANDARDIZE DATA
# ============================================
print("\nStandardizing data...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val)

# ============================================
# 3. CONVERT TO TENSORS (CNN expects 3D: [batch, channels, features])
# ============================================
X_train_t = torch.FloatTensor(X_train_scaled).unsqueeze(1)  # Add channel dimension
y_train_t = torch.FloatTensor(y_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled).unsqueeze(1)
y_test_t = torch.FloatTensor(y_test_scaled)
X_val_t = torch.FloatTensor(X_val_scaled).unsqueeze(1)
y_val_t = torch.FloatTensor(y_val_scaled)

# ============================================
# 4. CREATE DATALOADERS
# ============================================
batch_size = 64
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ============================================
# 5. DEFINE 1D-CNN MODEL
# ============================================
class CNN1D(nn.Module):
    def __init__(self, input_channels=1, input_features=21, output_features=30):
        super(CNN1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * input_features, 64),
            nn.ReLU(),
            nn.Linear(64, output_features)
        )
    
    def forward(self, x):
        return self.encoder(x)

model = CNN1D()
print(f"\nModel: {model}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 6. TRAINING SETUP
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================
# 7. TRAINING LOOP
# ============================================
print("\nTraining CNN...")
epochs = 50
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
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")

# ============================================
# 8. EVALUATE
# ============================================
print("\nEvaluating on test set...")
model.eval()
with torch.no_grad():
    X_test_t_device = X_test_t.to(device)
    y_pred_scaled = model(X_test_t_device).cpu().numpy()

# Inverse transform to get actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2))
print(f"Test RMSE: {rmse:.4f}")

# Per-channel RMSE
channel_rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2, axis=0))
print(f"Per-channel RMSE (first 5): {channel_rmse[:5]}")

# ============================================
# 9. PLOT RESULTS
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss curves
axes[0,0].plot(train_losses, label='Train Loss')
axes[0,0].plot(val_losses, label='Val Loss')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Training Curves')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Predicted vs actual (first 5 test samples)
for i in range(min(5, len(y_test_actual))):
    axes[0,1].plot(y_test_actual[i], 'b-', alpha=0.5, label='Actual' if i==0 else '')
    axes[0,1].plot(y_pred[i], 'r--', alpha=0.5, label='Predicted' if i==0 else '')
axes[0,1].set_xlabel('Channel')
axes[0,1].set_ylabel('Value')
axes[0,1].set_title('Predicted vs Actual (First 5 Test Samples)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Per-channel RMSE
axes[1,0].bar(range(len(channel_rmse)), channel_rmse)
axes[1,0].set_xlabel('Channel')
axes[1,0].set_ylabel('RMSE')
axes[1,0].set_title('RMSE per Output Channel')
axes[1,0].grid(True, alpha=0.3)

# Scatter plot
axes[1,1].scatter(y_test_actual.flatten(), y_pred.flatten(), alpha=0.1, s=1)
axes[1,1].plot([y_test_actual.min(), y_test_actual.max()], 
               [y_test_actual.min(), y_test_actual.max()], 'r--', linewidth=2)
axes[1,1].set_xlabel('Actual')
axes[1,1].set_ylabel('Predicted')
axes[1,1].set_title(f'Prediction Scatter (RMSE: {rmse:.4f})')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_results.png', dpi=150)
print("\n✅ Saved results to 'cnn_results.png'")
plt.show()

print(f"\nCNN Test RMSE: {rmse:.4f}")
