import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def euler_to_rotation_matrix(euler_angles):
    return Rotation.from_euler("xyz", euler_angles).as_matrix()


def reconstruct_volume(frames, transforms, probe_specs, voxel_size=0.5):
    translations = transforms[:, :3]
    rotations = [euler_to_rotation_matrix(euler) for euler in transforms[:, 3:]]

    depth = probe_specs["depth"]
    width = probe_specs["width"]
    marker_to_probe_bottom = probe_specs["marker_to_probe_bottom"]

    axial_scale = depth / frames.shape[1]
    lateral_scale = width / frames.shape[2]

    frame_corners = np.array(
        [[0, 0, 0], [width, 0, 0], [0, depth, 0], [width, depth, 0]]
    )
    transform_matrix = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    frame_corners = frame_corners @ transform_matrix.T
    frame_corners[:, 2] += marker_to_probe_bottom

    all_corners = (
        np.einsum("ijk,lk->ilj", rotations, frame_corners)
        + translations[:, np.newaxis, :]
    )
    min_corner = np.min(all_corners.reshape(-1, 3), axis=0)
    max_corner = np.max(all_corners.reshape(-1, 3), axis=0)

    volume_shape = np.ceil((max_corner - min_corner) / voxel_size).astype(int) + 1
    volume = np.zeros(volume_shape, dtype=np.float32)
    counts = np.zeros(volume_shape, dtype=np.int32)

    x, y = np.meshgrid(
        np.arange(frames.shape[2]) * lateral_scale,
        np.arange(frames.shape[1]) * axial_scale,
    )
    frame_coords = np.stack((x, y, np.zeros_like(x)), axis=-1)
    frame_coords = frame_coords @ transform_matrix.T
    frame_coords[:, :, 2] += marker_to_probe_bottom

    print("Reconstructing volume...")
    for frame, translation, rotation in tqdm(
        zip(frames, translations, rotations), total=len(frames)
    ):
        if frame.ndim == 3:
            frame = np.mean(frame, axis=-1).astype(np.float32)

        world_coords = np.einsum("ij,klj->kli", rotation, frame_coords) + translation
        voxel_coords = np.round((world_coords - min_corner) / voxel_size).astype(int)

        mask = np.all((voxel_coords >= 0) & (voxel_coords < volume_shape), axis=2)
        valid_voxels = voxel_coords[mask]
        valid_intensities = frame[mask]

        np.add.at(volume, tuple(valid_voxels.T), valid_intensities)
        np.add.at(counts, tuple(valid_voxels.T), 1)

    mask = counts > 0
    volume[mask] /= counts[mask]

    return volume, min_corner, voxel_size
