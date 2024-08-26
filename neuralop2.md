# Neural Operator for 3D Ultrasound Reconstruction from Freehand 2D Scans

## Input Spaces

1. Image Space: $\mathcal{I} = L^2(\Omega; \mathbb{R})$, where $\Omega \subset \mathbb{R}^2$ is the image domain
2. Pose Space: $\mathcal{P} = SE(3)$, the Special Euclidean group in 3D
3. Time Domain: $\mathcal{T} = [0, T]$

## Input Functions

1. Image Function: $I: \mathcal{T} \to \mathcal{I}$
   $I(t)$ represents the 2D grayscale ultrasound image at time $t$
2. Pose Function: $P: \mathcal{T} \to \mathcal{P}$
   $P(t)$ represents the 6-DoF pose (position and orientation) of the ultrasound probe at time $t$

## Output Space

3D Volume Space: $\mathcal{V} = L^2(\Omega_3; \mathbb{R})$, where $\Omega_3 \subset \mathbb{R}^3$ is the 3D volume domain

## Operator Definition

We aim to learn the operator $\mathcal{G}: (\mathcal{I} \times \mathcal{P})^\mathcal{T} \to \mathcal{V}$ such that:

$\mathcal{G}(I, P) = V$

where $V \in \mathcal{V}$ is the reconstructed 3D volume.

## Neural Operator Approximation

We approximate $\mathcal{G}$ with a neural operator $\mathcal{G}_\theta$:

$\mathcal{G}_\theta(I, P) \approx V$

## Discretization

In practice, we work with discretized versions:

1. $I_i = I(t_i) \in \mathbb{R}^{H \times W}$ for $i = 1, \ldots, N$, where $H$ and $W$ are image height and width
2. $P_i = P(t_i) \in \mathbb{R}^4 \times \mathbb{R}^3$ for $i = 1, \ldots, N$, representing rotation (as quaternion) and translation
3. $V \in \mathbb{R}^{D \times D \times D}$, where $D$ is the dimension of the reconstructed volume

## Modified Neural Operator Architecture

We can use a combination of convolutional layers for image processing and Fourier Neural Operator (FNO) layers for spatial integration:

1. Image Encoding: $\mathcal{E}: \mathbb{R}^{H \times W} \to \mathbb{R}^d$
2. Pose Embedding: $\mathcal{Q}: \mathbb{R}^7 \to \mathbb{R}^d$
3. Temporal-Spatial Integration: $\mathcal{F}: (\mathbb{R}^d \times \mathbb{R}^d)^N \to \mathbb{R}^{D \times D \times D}$

The neural operator $\mathcal{G}_\theta$ is defined as:

$\mathcal{G}_\theta(I, P) = \mathcal{F}(\{\mathcal{E}(I_i), \mathcal{Q}(P_i)\}_{i=1}^N)$

## Fourier Neural Operator Layer

For the temporal-spatial integration, we use FNO layers:

$\mathcal{F}_k(v)(x) = \sigma\left(W v(x) + \mathcal{F}^{-1}\left(R_k \cdot \mathcal{F}(v)\right)(x)\right)$

where $x \in \Omega_3$, and $\mathcal{F}$ is the 3D Fourier transform.

## Loss Function

We can use a combination of voxel-wise loss and structural similarity:

$\mathcal{L}(\theta) = \alpha \|V - V_{gt}\|_2^2 + \beta (1 - \text{SSIM}(V, V_{gt}))$

where $V_{gt}$ is the ground truth 3D volume, and SSIM is the Structural Similarity Index.

## Training Algorithm

```
Input: Training data {(I_j, P_j, V_j)}_{j=1}^M
Output: Trained neural operator $\mathcal{G}_\theta$

1: Initialize $\theta$ randomly
2: for epoch = 1 to num_epochs do
3:     for j = 1 to M do
4:         Compute $\hat{V}_j = \mathcal{G}_\theta(I_j, P_j)$
5:         Compute loss: $\mathcal{L}_j = \alpha \|\hat{V}_j - V_j\|_2^2 + \beta (1 - \text{SSIM}(\hat{V}_j, V_j))$
6:         Update $\theta$ using gradient descent on $\mathcal{L}_j$
7:     end for
8: end for
9: return $\mathcal{G}_\theta$
```

## Challenges and Considerations

1. Irregular Sampling: Freehand ultrasound results in non-uniform sampling of the 3D space
2. Interpolation: The network needs to learn to interpolate between sampled planes
3. Alignment: Accurate pose estimation is crucial for proper 3D reconstruction
4. Memory Constraints: Processing sequences of 2D images and reconstructing large 3D volumes can be memory-intensive
