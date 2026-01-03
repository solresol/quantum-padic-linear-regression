# Algorithm Summary: p-adic Linear Regression

This document summarizes the classical and quantum algorithms for p-adic linear regression based on the papers in `../papers/polynomial-approximation` and `../papers/quantum-padic-regression`.

## Background: p-adic Numbers and Distance

The **p-adic distance** measures divisibility by a prime p rather than Euclidean closeness.

For a prime p and non-zero rational x = a/b:
- **p-adic valuation**: v_p(x) = (highest power of p dividing a) - (highest power of p dividing b)
- **p-adic absolute value**: |x|_p = p^(-v_p(x))
- **p-adic distance**: d_p(a,b) = |a - b|_p

Two numbers are p-adically close if their difference is highly divisible by p. For example, 3-adically: 1 and 28 are very close (difference = 27 = 3^3).

### Key Property: Strong Triangle Inequality (Ultrametricity)

|x + y|_p ≤ max(|x|_p, |y|_p)

This is stronger than the normal triangle inequality and is crucial to both algorithms.

## The p-adic Linear Regression Problem

**Given**: Points (X_i, y_i) where X_i ∈ ℚ^n and y_i ∈ ℚ

**Find**: An affine function F: ℚ^n → ℚ minimizing Σ|F(X_i) - y_i|_p

This is fundamentally different from Euclidean least squares:
- **No single global minimum**: Multiple equally-optimal solutions may exist
- **Infinite local minima**: Adding any multiple of p^t to coefficients creates local minima
- **Gradient descent fails**: The loss function is locally constant almost everywhere (every p-adic ball is a plateau)

## Core Theorem (Hyperplane Intersection)

**Theorem**: An optimal affine function of n variables must pass through at least n+1 points in the dataset.

This is proven using the strong triangle inequality. The key insight is that if a hyperplane passes through fewer than n+1 points, we can always find an adjustment that:
1. Keeps the already-zero residuals at zero
2. Makes at least one more residual zero
3. Does not increase any other residual (due to ultrametricity)

## Classical Algorithm

### Brute Force Approach

Based on the core theorem, the classical algorithm is:

1. **Enumerate all (n+1)-subsets** of data points
2. For each subset, compute the unique hyperplane through those n+1 points
3. Calculate the p-adic residual sum for all other points
4. Return the hyperplane with minimum total residual

### Complexity

For a dataset of r points in n dimensions:
- Number of candidate hyperplanes: C(r, n+1) = O(r^(n+1))
- Residuals per hyperplane: O(r)
- **Total complexity**: O(r^(n+2))

For 1D (finding a line): O(r^3)

### Why Classical Can't Do Better

- No gradient signal (loss landscape is flat everywhere locally)
- Cannot use divide-and-conquer (no ordering that respects p-adic distance)
- Local minima exist at every multiple of every power of p

## Quantum Algorithm (Sketch)

The quantum algorithm exploits the "ladder structure" of p-adic numbers.

### Key Insight

If (m, b) is optimal, then (m + p^k, b + p^k) is also quite good. Local minima exist at:
- Every multiple of p
- Every multiple of p^2 (better)
- Every multiple of p^3 (even better)
- etc.

### Algorithm Outline

**Preprocessing**: Convert all values to integers (multiply by LCM of denominators). This is O(n^2).

**Define**: F(m, b) = sum of p-adic residuals for gradient m and intercept b

**Round 1** (find optimal values mod p):
1. Create superposition of all (m, b) pairs
2. Compute m' = m mod p and b' = b mod p
3. Compute m'' = m - m' and b'' = b - b'
4. For i, j ∈ {0, ..., p-1}: calculate R_(i,j) = F(m'' + i, b'' + j)
5. Identify (i, j) with lowest residual
6. Apply Fourier transform to collapse to optimal (i, j) mod p

**Round 2** (find optimal values mod p^2):
1. Create superposition of gradients ≡ i (mod p) and intercepts ≡ j (mod p)
2. Now find i', j' such that (i'p + i, j'p + j) is optimal mod p^2
3. Still only p^2 comparisons needed

**Round t** (find optimal values mod p^t):
Continue until p^t exceeds the maximum possible integer values

### Complexity

**O(n k p^2)** where:
- n = dataset size
- k = dimensionality
- p = prime number

Plus integer scaling factors from the preprocessing step (likely O(n log n)).

### Open Questions in the Quantum Paper

1. "In the classical case, you can't do this ladder up. Why does it work in the quantum case?"
   - The quantum superposition allows exploring all branches simultaneously

2. The integer conversion step is messy - can it be avoided?

3. Exact proof of quantum advantage needs formalization

## Comparison

| Aspect | Classical | Quantum |
|--------|-----------|---------|
| Complexity (1D) | O(r^3) | O(r p^2) |
| Complexity (n-D) | O(r^(n+2)) | O(r n p^2) |
| Approach | Enumerate all (n+1)-subsets | Ladder ascent via superposition |
| Main obstacle | Combinatorial explosion | Integer conversion overhead |

For small p and large datasets, the quantum algorithm offers significant speedup.

## Applications

1. **Hierarchical data**: Biological taxonomies, grammatical structures
2. **Polynomial approximation**: Degree as a distance metric
3. **Pedagogical**: Simpler than Shor's algorithm, good for teaching quantum computing

## Implementation Notes

To implement the quantum algorithm:

1. **Preprocessing circuit**: Convert rationals to integers
2. **Modular arithmetic**: Compute m mod p^t efficiently on quantum computer
3. **Residual oracle**: F(m, b) as a quantum oracle
4. **QFT**: Fourier transform for collapsing superposition
5. **Iteration control**: Loop until p^t > max integer value

The classical algorithm should be implemented first as a baseline for comparison.
