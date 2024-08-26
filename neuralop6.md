[Previous content remains the same]

## Approach 4: Skewer-to-Curve Transformation

### Formulation

Let $s \in [0, 1]$ be the parametric variable representing position along the skewer/curve.

Define the skewer (initial state) as:
$S(s) = (0, 0, sL)$, where $L$ is the length of the skewer.

We aim to find a parametric curve $C(s): [0, 1] \to \mathbb{R}^3$ such that:

$C(s) = (x(s), y(s), z(s))$

where $x(s)$, $y(s)$, and $z(s)$ are continuous functions with $frame\_count$ points of inflection.

### Constraints

1. $C(0) = (0, 0, 0)$ and $C(1) = (x_f, y_f, L)$, where $(x_f, y_f)$ is the final 2D displacement.
2. $\|C'(s)\| \approx L$ for all $s$ (approximate arc-length parameterization).
3. Minimize $\sum_{i=1}^{N-1} \|\text{OpticalFlow}(I_i \circ T(s_i), I_{i+1} \circ T(s_{i+1}))\|_2^2$,
   where $T(s_i)$ is the transformation that aligns frame $i$ to the curve at $s_i = i/(N-1)$.

### Algorithm

1. Initialize $C(s)$ as a straight line (equivalent to $S(s)$).
2. For each iteration:
   a. Compute transformations $T(s_i)$ to map frames onto $C(s)$.
   b. Calculate optical flow between transformed adjacent frames.
   c. Update $C(s)$ to minimize optical flow while maintaining constraints.
   d. Enforce smoothness and inflection point constraints on $C(s)$.
3. Final reconstruction: Project frames onto the optimized $C(s)$.

### Mermaid Chart

```mermaid
graph TD
    A[2D Image Sequence] --> B[Initialize Straight Skewer]
    B --> C[Compute Frame Transformations]
    C --> D[Calculate Optical Flow]
    D --> E[Update Curve C(s)]
    E --> F{Converged?}
    F -->|No| C
    F -->|Yes| G[Project Frames onto Curve]
    G --> H[Output: 3D Reconstruction]
```

### Mathematical Details

We can represent $C(s)$ using splines or a sum of basis functions:

$C(s) = \sum_{k=1}^K \alpha_k \phi_k(s)$

where $\phi_k(s)$ are basis functions (e.g., cubic splines) and $\alpha_k$ are coefficients to be optimized.

The optimization problem becomes:

$\min_{\{\alpha_k\}} \sum_{i=1}^{N-1} \|\text{OpticalFlow}(I_i \circ T(s_i), I_{i+1} \circ T(s_{i+1}))\|_2^2$

subject to:

1. $C(0) = (0, 0, 0)$ and $C(1) = (x_f, y_f, L)$
2. $\|\sum_{k=1}^K \alpha_k \phi_k'(s)\| \approx L$ for all $s$
3. $C(s)$ has $frame\_count$ points of inflection

This formulation allows us to transform the straight skewer of ultrasound frames into a curved 3D path that minimizes optical flow between adjacent frames, effectively reconstructing the 3D structure of the scanned tissue.
