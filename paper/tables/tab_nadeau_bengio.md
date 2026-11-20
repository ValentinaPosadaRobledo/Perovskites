Secondary comparisons (not confirmatory): PU pre-DFT vs. each baseline, TPR at P(ŷ = 1) = 0.10 per fold on the compositions scored by both; corrected resampled t-test (Nadeau and Bengio), J = seeds × folds with positives; Holm within each split; * p < 0.05. TPR is the mean over folds.

| Split   | Baseline              |   TPR PU |   TPR baseline | ΔTPR [95 % CI]          |   J | p (Holm)   |
|:--------|:----------------------|---------:|---------------:|:------------------------|----:|:-----------|
| random  | PU post-DFT           |    0.368 |          0.383 | -0.015 [-0.036, +0.006] |  50 | 0.162      |
| random  | E_hull only           |    0.368 |          0.301 | +0.067 [+0.024, +0.110] |  50 | 0.006 *    |
| random  | Symmetry only         |    0.368 |          0.296 | +0.071 [+0.034, +0.108] |  50 | < 0.001 *  |
| random  | τ < 4.18              |    0.301 |          0.182 | +0.120 [+0.065, +0.174] |  50 | < 0.001 *  |
| random  | t window (Bartel)     |    0.303 |          0.19  | +0.113 [+0.064, +0.162] |  50 | < 0.001 *  |
| random  | Charge balance (ICSD) |    0.431 |          0.143 | +0.288 [+0.245, +0.331] |  50 | < 0.001 *  |
| cluster | PU post-DFT           |    0.401 |          0.427 | -0.026 [-0.143, +0.090] |  41 | 1.000      |
| cluster | E_hull only           |    0.401 |          0.448 | -0.047 [-0.167, +0.073] |  41 | 1.000      |
| cluster | Symmetry only         |    0.401 |          0.383 | +0.018 [-0.126, +0.162] |  41 | 1.000      |
| cluster | τ < 4.18              |    0.287 |          0.198 | +0.089 [+0.005, +0.172] |  41 | 0.189      |
| cluster | t window (Bartel)     |    0.288 |          0.226 | +0.062 [-0.013, +0.136] |  41 | 0.416      |
| cluster | Charge balance (ICSD) |    0.438 |          0.194 | +0.244 [+0.065, +0.422] |  41 | 0.052      |
