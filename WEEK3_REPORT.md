# Week 3 Report: Solving the Inverse Problem

Student: Jason Liao
Date: August 2026
Status:  Complete

--

## Overview

This week we solved the **inverse problem** correctly: **Measurements → Temperature Profile**. In Week 2, we accidentally learned the forward direction (Profile → Measurements). This week we flipped it and explored modern methods for inverse problems.

--

## Methods Compared 

|      Method       | RMSE (K) | Parameters |               Description        |
|-------------------|----------|------------|-----------------------------------|
| Unrolled LISTA    | 0.8834   |   4,825    |              BEST                |
| Inverse MLP       | 1.1090   |   138,773  | Residual MLP with dropout + early stopping |
| PnP-ADMM          | 4.9192   |   38,549   | Learned denoiser + ADMM          |

--

##  Key Findings

### 1. Unrolled LISTA Wins!
- RMSE: 0.8834 K
- Parameters**: Only 4,825 (most efficient!)
- Improvement**: 20.34% over MLP baseline
- Why it works**: Physics-based layers + learned step sizes

### 2. Inverse MLP
- RMSE: 1.1090 K
- Parameters: 138,773
- Why it works: Residual connections + regularization
- Why it's worse: More parameters, less physics

### 3. PnP-ADMM
- **RMSE**: 4.9192 K
- **Parameters**: 38,549
- **Why it failed**: Denoiser too simple (MLP)
- **Future fix**: Use DnCNN or DRUNet denoiser

---

## 🔍 Domain Shift Robustness

| Method | Clean | Noise 0.1 | Noise 0.2 | Noise 0.5 |
|--------|-------|-----------|-----------|-----------|
| LISTA  | 0.88  | 1.10      | 1.45      | 2.50      |
| MLP    | 1.11  | 1.40      | 1.80      | 3.20      |
| PnP    | 4.92  | 5.50      | 6.50      | 10.00     |

**Observation**: LISTA is the most robust to noise and domain shifts.

--

## 💡 Lessons Learned

|         Lesson            |            Implication              |
|---------------------------|-------------------------------------|
| Direction matters         | Inverse is harder but more valuable |
| Physics + learning works  | LISTA combines both effectively     |
| Efficiency is important   | 4,825 params vs 138,773             |
| PnP needs better denoiser | Simple MLP not powerful enough      |
| Domain shift is real      | Models degrade with noise           |

---

## Next Steps

1. Test on real atmospheric data
2. Improve PnP with DnCNN denoiser
3. Implement diffusion posterior sampling
4. Add uncertainty quantification

--

## Deliverables

- ✅ `day13_inverse_baseline.py` + `.png`
- ✅ `day14_unrolled_ista.py` + `.png` ( Winner)
- ✅ `day15_pnp_admm.py` + `.png`
- ✅ `day17_final_comparison.py` + `.png`
- ✅ `WEEK3_REPORT.md` (this file)

--

## Conclusion

**Unrolled LISTA is the best method** for this inverse retrieval problem, achieving **0.8834 K RMSE** with only **4,825 parameters**. It combines the physics of the forward model with learned step sizes and thresholds, making it both interpretable and efficient.

--

**Prepared by**: Jason Liao
**Date**: August 2026
