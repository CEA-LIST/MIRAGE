import numpy as np


def pointcloud_to_spatiotemporal_voxel(points, stacks, LSTM=False):
    """
    Convert point cloud with features into a spatiotemporal voxel grid.

    Args:
        points: numpy array of shape (N, 4 + F)
                columns: (x, y, z, t, f1, f2, ...)*
        stacks: integer, stack length of the point cloud

    Returns:
        voxel_grid: numpy array of shape (X, Y, Z, T, C)
                    C = 1 + F
                    channel 0 = occupancy count
                    channel 1..F = mean feature value per voxel
    """
    assert points.ndim == 2 and points.shape[1] >= 4
    no_time = False
    if LSTM:
        if np.max(points[:,3]) == 0: # not using time
            grid_size = (32, 32, 10, 1)
            no_time = True
        else:
            ## with making superframes from single frames (otherwise too sparse)
            superframe = (stacks // 5)
            grid_size = (32, 32, 10, superframe)
            points[:,3] = points[:,3] * superframe
            ## legacy
            # grid_size = (32, 32, 10, stacks)
    else:
        # for time as a feature : for CNN_tiny (not using LSTM)#
        no_time = True 
        grid_size = (32, 32, 10, 1)

    X, Y, Z, T = grid_size

    min_bound = np.array([np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2]), np.min(points[:,3])])
    max_bound = np.array([np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2]), np.max(points[:,3])])

    voxel_size = (max_bound - min_bound) / np.array(grid_size)
    if no_time:
        voxel_size[3] = 1

    # --- split coordinates and features ---
    coords = points[:, :4]
    if LSTM:
        feats  = points[:, 4:] if points.shape[1] > 4 else None
    else:
        feats  = points[:, 3:] if points.shape[1] > 3 else None # use time as a feature
    F = 0 if feats is None else feats.shape[1]

    # --- compute voxel indices ---
    voxel_coords = np.floor((coords - min_bound) / voxel_size).astype(np.int64)

    valid_mask = (
        (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < X) &
        (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < Y) &
        (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < Z) &
        (voxel_coords[:, 3] >= 0) & (voxel_coords[:, 3] < T)
    )

    voxel_coords = voxel_coords[valid_mask]
    if voxel_coords.size == 0:
        C = 1 + F
        return np.zeros((X, Y, Z, T, C), dtype=np.float32)

    if feats is not None:
        feats = feats[valid_mask]

    # --- flatten voxel indices ---
    linear_idx = np.ravel_multi_index(
        voxel_coords.T, dims=(X, Y, Z, T)
    )

    num_voxels = X * Y * Z * T

    # --- occupancy ---
    counts = np.bincount(
        linear_idx, minlength=num_voxels
    ).astype(np.float32)

    # --- feature accumulation ---
    channels = [counts]

    if feats is not None:
        for f in range(F):
            feat_sum = np.bincount(
                linear_idx,
                weights=feats[:, f],
                minlength=num_voxels
            )
            mean_feat = np.zeros_like(feat_sum, dtype=np.float32)
            nonzero = counts > 0
            mean_feat[nonzero] = feat_sum[nonzero] / counts[nonzero]
            channels.append(mean_feat.astype(np.float32))

    voxel_grid = np.stack(channels, axis=-1)
    voxel_grid = voxel_grid.reshape(X, Y, Z, T, 1 + F)

    return voxel_grid