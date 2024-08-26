# Comprehensive Compendium of 3D Ultrasound Reconstruction Approaches

## Notation and Definitions

- $I(t): [0, T] \to \mathbb{R}^{H \times W}$: 2D grayscale ultrasound image function
- $P(t): [0, T] \to SE(3)$: probe pose function
- $\{(I_i, P_i)\}_{i=1}^N$: discrete set of image-pose pairs
- $\Omega \subset \mathbb{R}^3$: 3D reconstruction space
- $V: \Omega \to \mathbb{R}$: reconstructed 3D volume
- $\mathcal{P} = \{(x, y, z, i) | (x, y, z) \in \mathbb{R}^3, i \in \mathbb{R}\}$: point cloud space

[Previous approaches 1-5 remain the same]

## Approach 6: Dense Matching for Ultrasound Frames

### Formulation

For each pair of consecutive frames $(I_i, I_{i+1})$, find a dense correspondence field $F_i: \mathbb{R}^2 \to \mathbb{R}^2$ such that:

$I_i(x, y) \approx I_{i+1}(F_i(x, y))$ for all $(x, y)$

### Algorithm

1. For each pair of consecutive frames $(I_i, I_{i+1})$:
   a. Initialize $F_i(x, y) = (x, y)$
   b. Repeat until convergence:
   i. Compute the error image: $E_i(x, y) = I_i(x, y) - I_{i+1}(F_i(x, y))$
   ii. Estimate update field $\Delta F_i$ using normalized cross-correlation on $E_i$
   iii. Update $F_i(x, y) \leftarrow F_i(x, y) + \lambda \Delta F_i(x, y)$
   iv. Regularize $F_i$ using bilateral filtering
2. Accumulate transformations: $T_i = F_1 \circ F_2 \circ ... \circ F_{i-1}$
3. Reconstruct 3D volume:
   For each voxel $(x, y, z)$ in $V$:
   a. Find $i$ such that $z \in [i\Delta z, (i+1)\Delta z]$
   b. Compute $(x', y') = T_i^{-1}(x, y)$
   c. Set $V(x, y, z) = I_i(x', y')$
