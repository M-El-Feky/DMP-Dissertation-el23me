import numpy as np

def test_dmp_performance(
    original_path,
    DMP_path,
    obstacle_paths=None,
    obstacle_positions=None
):

    original_path = np.asarray(original_path, dtype=float)
    DMP_path = np.asarray(DMP_path, dtype=float)

    assert original_path.ndim == 2 and DMP_path.ndim == 2
    assert original_path.shape == DMP_path.shape

    d, T = DMP_path.shape

    final_goal_error = float( np.linalg.norm(DMP_path[:, -1] - original_path[:, -1]))

    point_error = np.linalg.norm(DMP_path - original_path, axis=0)  # (T,)
    RMS_tracking_error = float(np.sqrt(np.mean(point_error ** 2)))

    deviation_mean = float(np.mean(point_error))
    deviation_RMS  = RMS_tracking_error
    deviation_max  = float(np.max(point_error))

    def path_length(path):
        return float(np.sum(np.linalg.norm(np.diff(path, axis=1), axis=0)))

    original_path_length = path_length(original_path)
    DMP_path_length = path_length(DMP_path)

    min_distance_each_obstacle = []

    if obstacle_paths is not None:
        if isinstance(obstacle_paths, np.ndarray): # 
            obstacle_paths = [obstacle_paths]
        else:
            obstacle_paths = list(obstacle_paths)

        for k, obs in enumerate(obstacle_paths):
            obs = np.asarray(obs, dtype=float)
            assert obs.shape == (d, T), f"obstacle_paths[{k}] must be shape {(d, T)}"

            valid = np.all(np.isfinite(obs), axis=0)  # ignore NaN (pre-spawn)

            if not np.any(valid):
                min_distance_each_obstacle.append(np.inf)
            else:
                dist = np.linalg.norm(DMP_path[:, valid] - obs[:, valid], axis=0)
                min_distance_each_obstacle.append(float(np.min(dist)))

    if obstacle_positions is not None:
        for k, obs in enumerate(obstacle_positions):
            obs = np.asarray(obs, dtype=float).reshape(d,)
            dist = np.linalg.norm(DMP_path - obs.reshape(d, 1), axis=0)
            min_distance_each_obstacle.append(float(np.min(dist)))

    # Overall min distance (None if no obstacles provided)
    if len(min_distance_each_obstacle) == 0:
        min_distance_overall = None
    else:
        min_distance_overall = float(np.min(min_distance_each_obstacle))

    metrics = {
        "final_goal_error": final_goal_error,
        "RMS_tracking_error": RMS_tracking_error,
        "min_distance_each_obstacle": min_distance_each_obstacle,
        "min_distance_overall": min_distance_overall,
        "original_path_length": original_path_length,
        "DMP_path_length": DMP_path_length,
    }

    return point_error, metrics