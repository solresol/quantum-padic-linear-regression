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

## Quantum Algorithm

The quantum algorithm exploits the "ladder structure" of p-adic numbers.

### Key Insight: Local Minima Structure

p-adic linear regression solutions have a regular structure of local minima:
- Local minima occur at every multiple of p
- Better local minima occur at every multiple of p²
- Even better local minima occur at every multiple of p³
- etc.

This means that if a* is the globally optimal coefficient, then a* mod p is likely to be
the locally optimal last digit within most "brackets" of p consecutive values.

### Algorithm Details (1D Case)

Assume we seek a coefficient 0 < a < p^k for some reasonable k (e.g., k=100).

**Round 1** (find optimal value mod p, i.e., the first p-adic digit):

1. **Prepare superposition**: Create quantum register in uniform superposition over all
   possible coefficient values:
   ```
   |ψ⟩ = (1/√N) Σ_{a=0}^{p^k - 1} |a⟩
   ```

2. **Compute bracket values**: For each value a in superposition, quantumly compute:
   - a_m = a mod p (the last digit in base p)
   - a_floor = a - a_m (lowest multiple of p ≤ a, i.e., same higher digits, last digit = 0)
   - a_ceil = a_floor + p - 1 (same higher digits, last digit = p-1)

   The "bracket" {a_floor, a_floor+1, ..., a_ceil} contains all p values that share
   the same higher digits as a but differ only in the last digit.

3. **Compare residuals within bracket**: Quantumly compute the residual sum F(a) and
   compare against F(a_floor), F(a_floor+1), ..., F(a_ceil). If a gives the minimum
   residual among all p candidates in its bracket, set marker s = 1, otherwise s = 0.

4. **Uncompute temporaries**: Erase a_m, a_floor, a_ceil (reverse the computation to
   disentangle these registers).

5. **Apply QFT**: Apply the Quantum Fourier Transform to the a register.

6. **Measure**: Observe the system. Due to the p-adic ladder structure, amplitudes
   reinforce for states where a mod p equals the globally optimal last digit. The
   measured value a', when reduced mod p, gives the first p-adic digit d₀.

**Round 2** (find optimal value mod p², i.e., the second p-adic digit):

Now we know a* ≡ d₀ (mod p). We repeat the process but with larger brackets:

1. **Prepare superposition**: Same as before, all values 0 to p^k - 1.

2. **Compute bracket values**:
   - a_floor = largest multiple of p² that is ≤ a
   - a_ceil = smallest multiple of p² that is > a, minus 1

   The bracket now contains p values: {a_floor, a_floor + p, a_floor + 2p, ..., a_ceil}
   These are all values that share the same digits beyond the second position.

3. **Compare residuals**: Check if a gives minimum residual among its bracket of p
   candidates (stepping by p, not by 1).

4. **Uncompute, QFT, Measure**: Same as round 1. The result mod p² gives the first
   two p-adic digits.

**Round t** (find optimal value mod p^t):

Continue similarly:
- Brackets step by p^(t-1)
- Compare among p candidates in each bracket: {a_floor, a_floor + p^(t-1), a_floor + 2·p^(t-1), ...}
- Measure and extract a mod p^t

Continue until p^t exceeds the maximum possible coefficient value.

### Why This Works

The quantum speedup comes from two key properties:

1. **Superposition over all values**: We prepare a superposition over the entire
   search space, not just p² values per round. Each computational basis state
   |a⟩ represents a complete candidate coefficient.

2. **Amplitude reinforcement via QFT**: The p-adic ladder structure means that
   local minima repeat every p (then every p², etc.). When we mark states that are
   locally optimal within their bracket and apply QFT, states sharing the correct
   p-adic digit interfere constructively. States with incorrect digits interfere
   destructively on average.

The crucial insight is that the *globally* optimal solution has *locally* optimal
digits at each position (with high probability), so the marking and QFT steps
preferentially amplify the correct solution.

### Extension to Multiple Variables

For 2D regression (finding m and b), we use quantum registers for both variables:
- |a⟩ → |m⟩|b⟩
- Brackets become 2D: compare among p² candidates
- Same QFT + measurement approach

For n-D regression:
- n+1 coefficient registers
- Compare among p^(n+1) candidates per bracket
- Complexity grows polynomially with dimension

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

### Required Quantum Components

To implement the full quantum algorithm:

1. **Large superposition preparation**: Create uniform superposition over all p^k possible
   coefficient values (not just p² per round).

2. **Quantum modular arithmetic**:
   - Compute a mod p^t
   - Compute a_floor = a - (a mod p^t)
   - These must be reversible for uncomputation

3. **Quantum residual oracle**: Compute F(a) = Σ|y_i - a·x_i|_p for all data points.
   This requires:
   - Quantum multiplication (a · x_i)
   - Quantum subtraction (y_i - a·x_i)
   - Quantum p-adic valuation (trailing zeros counting)
   - Quantum addition (accumulate valuations)

4. **Quantum minimum finding within bracket**: Compare F(a) against F(a_floor),
   F(a_floor + step), ..., F(a_ceil) and set marker s=1 if a is minimum.
   This requires computing p residual sums and finding the minimum.

5. **Uncomputation**: Reverse all temporary computations to disentangle auxiliary
   registers before QFT.

6. **QFT**: Quantum Fourier Transform on the coefficient register.

7. **Iteration control**: Repeat for t = 1, 2, ... until p^t > max coefficient value.

### Current Implementation Status

The current `quantum_ladder.py` implements a simplified version that:
- Uses classical computation for valuation sums (not truly quantum)
- Uses amplitude state preparation based on classically computed valuations
- Does not implement the full superposition + bracket comparison + QFT approach

A correct implementation would require:
- Much larger quantum circuits (superposition over p^k states, not p²)
- Full quantum arithmetic for residual computation
- Proper uncomputation to enable QFT interference
- The quantum minimum-finding subroutine for bracket comparisons

### Building Blocks Available

The repository includes quantum arithmetic primitives that could be used:
- `quantum_arithmetic.py`: Addition, subtraction, multiplication by constant
- `twoadic.py`: Trailing zeros counting (p-adic valuation for p=2)
- `initialise.py`: Loading classical values into quantum registers

These would need to be composed into the full algorithm with proper uncomputation.
