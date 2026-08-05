# Week 2 Project: Atmospheric Profile Retrieval

## Overview
This project compares different neural network architectures for retrieving atmospheric temperature and dew-point profiles from filter power measurements. The task is an **inverse problem** - going from measurements back to the underlying atmospheric state.

## Dataset
- **Samples**: 150,025 training, 221 test, 221 validation
- **Input**: 21 vertical levels (temperature profile)
- **Output**: 30 channels (PC coefficients for reconstruction)
- **Synthetic data** based on realistic atmospheric profiles

## Models Compared

### 1. Plain MLP
- Simple feedforward neural network
- 8 layers, 64 hidden units
- No skip connections
- Parameters: 32,478

### 2. 1D-CNN
- Convolutional layers along filter axis
- 3 convolutional layers (32, 64, 128 channels)
- Captures local structure in the filter ordering
- Parameters: 205,086

### 3. Residual MLP
- MLP with skip connections
- 8 layers with residual blocks
- Skip connections help gradients flow through the network
- Parameters: 36,638

## Results

| Model | RMSE | Parameters | Improvement |
|-------|------|------------|-------------|
| **Residual MLP** | **3.0591** | 36,638 | **Best** |
| 1D-CNN | 3.0970 | 205,086 | 1.24% worse |
| Plain MLP | 3.2452 | 32,478 | 6.08% worse |

## Discussion

### What Worked
1. **Residual connections**: Skip connections significantly improved performance, showing they help deeper networks train better
2. **CNN architecture**: Despite having more parameters, the CNN performed well, capturing local structure
3. **PCA preprocessing**: Using PCA coefficients as targets made the problem tractable

### What Didn't Work
1. **Plain MLP**: Deeper networks without skip connections struggled to learn
2. **The CNN** didn't outperform the residual MLP despite having more parameters

### Key Insights
1. **Skip connections are crucial** for training deeper networks
2. **More parameters don't guarantee better performance** - the residual MLP had far fewer parameters (36,638) but performed best
3. **The problem benefits from architectural inductive bias** - both CNNs and residual connections help

## Error Analysis
- Temperature reconstruction: RMSE ~3.06-3.25
- Upper atmosphere layers showed larger errors
- Dew-point reconstruction is more challenging than temperature

## Conclusions
The Residual MLP achieved the best performance with RMSE of 3.0591, demonstrating the value of skip connections for this retrieval task. The CNN showed competitive performance but couldn't beat the residual architecture. These results provide a solid baseline for future work with real atmospheric data.

## Future Work
- [ ] Test on real atmospheric data
- [ ] Experiment with attention mechanisms
- [ ] Try different PCA components (k)
- [ ] Explore noise sensitivity experiments

## Files
- `cnn_results.png` - CNN training and evaluation plots
- `resnet_comparison.png` - ResNet vs Plain MLP comparison
- `model_comparison.png` - Overall model comparison
- `q6_phase_a_dataset.h5` - Dataset
- `cnn_model_fixed.py` - CNN implementation
- `resnet_model.py` - ResNet implementation
- `full_comparison.py` - Comparison script

## Week 2 Complete ✅
