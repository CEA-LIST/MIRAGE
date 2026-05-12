import logging
import numpy as np
import torch


def random_translation(points, kps=None, translation_range=0.3):
    """
    Apply random translation to the point cloud.

    Parameters:
    points (np.ndarray): Point cloud data of shape (N, 3) where N is the number of points.
    translation_range (float): Range for random translation.

    Returns:
    np.ndarray: Translated point cloud data.
    """
    translation = np.random.uniform(-translation_range, translation_range, size=(1, 3))
    points[:, :3] += translation
    if kps is not None:
        kps[:, :3] += translation
    return points, kps


def random_rotation(points, kps=None, rotation_range=np.pi * 30 / 180):
    """
    Apply a random rotation around one randomly chosen axis (X, Y, or Z) to the point cloud.

    Parameters:
    points (np.ndarray): Point cloud data of shape (N, 3) where N is the number of points.
    rotation_range (float): Maximum rotation angle (in radians) around the chosen axis.

    Returns:
    np.ndarray: Rotated point cloud data.
    """
    # Randomly choose an axis: 0 = X, 1 = Y, 2 = Z
    axis = np.random.choice(['x', 'y', 'z']) #TODO
    # axis = 'y'
    theta = np.random.uniform(-rotation_range, rotation_range)

    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        R = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    else:  # 'z'
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])

    # Apply the rotation
    points[:, :3] = np.dot(points[:, :3], R.T)
    if kps is not None:
        kps[:, :3] = np.dot(kps[:, :3], R.T)
        
    if points.shape[1] > 5: # apply rotation to radial velocity (corresponds to "x" axis)
        zeros_array = np.zeros((points[:, 5].shape[0], 2))
        expanded_array = points[:, 5][:, np.newaxis]
        velocity = np.concatenate((expanded_array, zeros_array), axis=1) # make it 3D vector (radial velocity, 0,0)
        velocity = np.dot(velocity, R.T)
        points[:, 5] = velocity[:, 0]

    return points, kps


def random_scaling(points, kps=None, scale_range=(0.8, 1.2)):
    """
    Apply random scaling to the point cloud.

    Parameters:
    points (np.ndarray): Point cloud data of shape (N, 3) where N is the number of points.
    scale_range (tuple): Range for random scaling.

    Returns:
    np.ndarray: Scaled point cloud data.
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= scale
    if kps is not None:
        kps[:, :3] *= scale
    return points, kps


def time_stretching(points, kps=None):
    """
    Apply random time stretching to the point cloud.

    This function modifies the fourth column (index 3) of the input point cloud,
    which is assumed to represent time or a similar sequential index. The values
    are scaled by a random factor, rounded to the nearest integer, and points
    falling outside the range [0, 1] are removed. Assumes time is rescaled between 0 and 1.

    Parameters:
    points (np.ndarray): Point cloud data of shape (N, D) where N is the number of points
                         and D is the dimensionality of each point. The fourth column
                         (index 3) is assumed to represent time.

    Returns:
    np.ndarray: Point cloud data with time values stretched and filtered.
    """
    stretch_factor = np.random.uniform(0.8, 1.2)
    points[:, 3] = points[:, 3] * stretch_factor
    points = points[(points[:, 3] >= 0) & (points[:, 3] <= 1)]
    return points, kps


def random_node_dropout(points, kps=None, dropout_rate=0.3):
    """
    Apply random node dropout to the point cloud.

    Args:
        points (np.ndarray): Point cloud data of shape (N, D).
        dropout_rate (float): Rate of node dropout.

    Returns:
        np.ndarray: Point cloud data with random nodes dropped.
    """
    num_points_to_drop = np.random.randint(0, int(points.shape[0] * dropout_rate) + 1)
    dropout_mask = np.ones(points.shape[0], dtype=bool)
    dropout_indices = np.random.choice(points.shape[0], num_points_to_drop, replace=False)
    dropout_mask[dropout_indices] = False
    points = points[dropout_mask]
    return points, kps


def add_random_points(points, kps=None, percentage=0.3):
    """
    Add random points to the point cloud.

    Args:
        points (np.ndarray): Point cloud data of shape (N, D), where D is either 3 or 4.
        percentage (float): Percentage of random points to add.

    Returns:
        np.ndarray: Point cloud data with added random points.
    """
    num_points_to_add = np.random.randint(0, int(points.shape[0] * percentage) + 1)
    if num_points_to_add == 0:
        return points, kps

    # Generate random points within the specified ranges
    random_points = np.zeros((num_points_to_add, points.shape[1]))

    # use the current point cloud to determine value ranges
    min_vals, max_vals = np.min(points, axis=0), np.max(points, axis=0)

    for dim in range(points.shape[1]):
        random_points[:, dim] = np.random.uniform(min_vals[dim], max_vals[dim], size=num_points_to_add)

    # Add the random points to the original point cloud
    points = np.vstack((points, random_points))
    if points.shape[1] > 3: # if it has time, reorder the point cloud by time coordinate
        points = points[np.argsort(points[:, 3])]
    return points, kps