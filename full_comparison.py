import os
os.environ['NUMPY_EXPERIMENTAL_DTYPE_API'] = '1'

import numpy as np
np._ARRAY_API = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Results from your trained models
results = {
    'Plain MLP': {
        'rmse': 3.2452,
        'params': 0,  # We'll calculate this
        'description': '8-layer feedforward network'
    },
    '1D-CNN': {
        'rmse': 3.0970,
        'params': 205086,
        'description': '3-layer convolutional network'
    },
    'Residual MLP': {
        'rmse': 3.0591,
        'params': 0,  # We'll calculate this
        'description': '8-layer network with skip connections'
    }
}

# Calculate parameter counts for MLP models
def calculate_mlp_params(input_dim=21, hidden_dim=64, output_dim=30, num_layers=8):
    # Plain MLP: input->hidden + (num_layers-1) hidden->hidden + hidden->output
    params = input_dim * hidden_dim + hidden_dim  # input layer with bias
    for _ in range(num_layers - 1):
        params += hidden_dim * hidden_dim + hidden_dim  # hidden layers with bias
    params += hidden_dim * output_dim + output_dim  # output layer with bias
    return params

def calculate_resnet_params(input_dim=21, hidden_dim=64, output_dim=30, num_blocks=4):
    # Residual MLP: input->hidden + num_blocks * (hidden->hidden + hidden->hidden) + hidden->output
    params = input_dim * hidden_dim + hidden_dim  # input layer with bias
    for _ in range(num_blocks):
        params += (hidden_dim * hidden_dim + hidden_dim)  # first linear layer
        params += (hidden_dim * hidden_dim + hidden_dim)  # second linear layer
    params += hidden_dim * output_dim + output_dim  # output layer with bias
    return params

# Update parameter counts
results['Plain MLP']['params'] = calculate_mlp_params()
results['Residual MLP']['params'] = calculate_resnet_params()

# Print comparison table
print("="*70)
print("MODEL COMPARISON RESULTS - WEEK 2")
print("="*70)
print(f"\n{'Model':<20} {'RMSE':<12} {'Parameters':<15} {'Improvement':<15}")
print("-"*70)

# Find best RMSE for improvement calculation
best_rmse = min([r['rmse'] for r in results.values()])

for name, data in results.items():
    improvement = ((data['rmse'] - best_rmse) / best_rmse) * 100
    if data['rmse'] == best_rmse:
        improvement_str = "Best"
    else:
        improvement_str = f"{improvement:.2f}% worse"
    print(f"{name:<20} {data['rmse']:.4f}    {data['params']:,}       {improvement_str}")

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print("1. ✅ Residual MLP performs best (RMSE: 3.0591)")
print(f"2. 📈 Residual MLP improves over Plain MLP by {((3.2452 - 3.0591)/3.2452*100):.2f}%")
print(f"3. 📈 CNN improves over Plain MLP by {((3.2452 - 3.0970)/3.2452*100):.2f}%")
print("4. 🎯 Skip connections help deeper networks train better")
print("5. 📊 CNN captures local structure but doesn't beat residual connections")
print("="*70)

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = list(results.keys())
rmse_values = [results[m]['rmse'] for m in models]
param_values = [results[m]['params'] for m in models]
colors = ['orange', 'blue', 'green']

# RMSE comparison
bars1 = axes[0].bar(models, rmse_values, color=colors)
axes[0].set_ylabel('RMSE')
axes[0].set_title('Model Performance Comparison')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, max(rmse_values) * 1.1)
for bar, val in zip(bars1, rmse_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom')

# Parameter count comparison
bars2 = axes[1].bar(models, param_values, color=colors)
axes[1].set_ylabel('Number of Parameters')
axes[1].set_title('Model Size Comparison')
axes[1].grid(True, alpha=0.3)
for bar, val in zip(bars2, param_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                 f'{val:,}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
print("\n✅ Saved comparison plot to 'model_comparison.png'")
print("="*70)
