# Distilled Unique Techniques in 3D Ultrasound Reconstruction

## Image Processing and Feature Extraction

1. FAST (Features from Accelerated Segment Test)
2. SIFT (Scale-Invariant Feature Transform)
3. SURF (Speeded Up Robust Features)
4. Normalized Cross-Correlation (NCC)

## Motion Estimation

5. Optical Flow Computation
6. Feature Matching (e.g., FLANN Matcher)
7. PnP (Perspective-n-Point) Algorithm
8. ICP (Iterative Closest Point)

## Pose Representation and Estimation

9. SE(3) Pose Representation
10. IMU Integration for Pose Prediction

## 3D Reconstruction Techniques

11. Back-projection of 2D Points to 3D
12. Triangulation of Matched Features
13. Voxel Grid Representation
14. TSDF (Truncated Signed Distance Function) Fusion

## Optimization Techniques

15. Bundle Adjustment (Local and Global)
16. Gradient Descent Optimization
17. Constrained Optimization (e.g., Sequential Quadratic Programming)

## Neural Network Approaches

18. Image Encoding Networks
19. Pose Embedding Networks
20. Temporal-Spatial Integration Networks
21. Point Cloud Generation Networks

## Mathematical Transformations

22. Fourier Transform (in Fourier Neural Operator)
23. **Skewer-to-Curve Transformation** [Highlighted: User-originated]
24. **4D to 3D Volume Transformation** [Highlighted: User-originated]

## Curve and Surface Representations

25. Parametric Curve Representation (e.g., B-splines)
26. **Inflection Point Constrained Curves** [Highlighted: User-originated]

## Multi-modal Data Processing

27. Pressure-based Depth Estimation
28. Sensor Fusion (IMU, Pressure, Image)

## Point Cloud Processing

29. Statistical Outlier Removal
30. Voxel Grid Filtering

## Specialized Techniques

31. **Dense Matching for Ultrasound Frames** [Highlighted: User-originated]
32. **Time as a Spatial Dimension** [Highlighted: User-originated]
33. Visual SLAM-inspired Mapping and Tracking

## Error Metrics and Loss Functions

34. Chamfer Distance (for Point Clouds)
35. Structural Similarity Index (SSIM)
36. Photometric Error

## Regularization Techniques

37. Bilateral Filtering
38. Smoothness Constraints on Transformations

## Computational Techniques

39. Parallel Processing of Frame Pairs
40. GPU Acceleration (implied for neural network approaches)
