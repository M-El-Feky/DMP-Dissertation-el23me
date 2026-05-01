import numpy as np

# DMP OA Function

def dmp_discrete_3d_moving_OA(
    pos, 
    dt, 
    obs_paths,
    kp=150.0, 
    kv=25.0, 
    alpha=5.0, 
    basis_num=40,
    speed=1.0,
    start_shift=None, 
    goal_shift=None,
    init_vel_scale=1.0, 
    init_vel_offset=None,
    oa_gamma=200.0, 
    oa_beta=10.0
):
    pos = np.asarray(pos, dtype=float)
    assert pos.ndim == 2 and pos.shape[0] == 3, "pos must be shape (3, T)"
    assert speed > 0, "speed must be > 0"

    T = pos.shape[1]
    tau_demo = dt * T
    tau_new  = tau_demo / speed
    T_new = int(np.round(tau_new / dt))
    T_new = max(T_new, 2)

    if obs_paths is None:
        obs_paths = []
    elif isinstance(obs_paths, np.ndarray): # single obstacle (3, T_new)
        obs_paths = [obs_paths]

    for k, p in enumerate(obs_paths):
        p = np.asarray(p, dtype=float)
        assert p.shape == (3, T_new), f"Obstacle {k} must be shape (3, T_new={T_new})"
        obs_paths[k] = p

    vel = np.gradient(pos, axis=-1) / dt
    acc = np.gradient(vel, axis=-1) / dt

    demo_start = pos[:, 0]
    demo_goal  = pos[:, -1]

    if start_shift is None: 
        start_shift = np.zeros(3)
    if goal_shift  is None: 
        goal_shift  = np.zeros(3)
    start_shift = np.asarray(start_shift, dtype=float).reshape(3,)
    goal_shift  = np.asarray(goal_shift, dtype=float).reshape(3,)

    newStart = demo_start + start_shift
    newGoal  = demo_goal  + goal_shift

    if init_vel_offset is None:
        init_vel_offset = np.zeros(3)
    init_vel_offset = np.asarray(init_vel_offset, dtype=float).reshape(3,)

    u = np.zeros(basis_num)
    c = np.zeros(basis_num)
    h = np.zeros(basis_num)

    for i in range(basis_num):
        u[i] = (1.0 / basis_num) * i
        c[i] = np.exp(-alpha * u[i])

    for i in range(basis_num - 1):
        h[i] = 0.5 / ((0.65 * (c[i+1] - c[i]))**2)
    h[basis_num - 1] = h[basis_num - 2]

    
    Phi = np.zeros(basis_num)
    Phi_total = np.zeros((T, basis_num))
    force = np.zeros((T, 3))

    s = 1.0
    for i in range(T):
        s = -alpha * s * dt / tau_demo + s

        addsum = 0.0
        for b in range(basis_num):
            Phi[b] = np.exp(-h[b] * (s - c[b])**2)
            addsum += Phi[b]

        Phi = Phi / addsum * s
        Phi_total[i, :] = Phi

        force[i, :] = (tau_demo*tau_demo)*acc[:, i] - kp*(demo_goal - pos[:, i]) + tau_demo*kv*vel[:, i]

    W = np.matmul(np.linalg.pinv(Phi_total), force)  # (basis_num, 3)

    DMP_pos = np.zeros((3, T_new))
    DMP_vel = np.zeros((3, T_new))
    DMP_acc = np.zeros((3, T_new))

    DMP_pos[:, 0] = newStart
    DMP_vel[:, 0] = init_vel_scale * vel[:, 0] + init_vel_offset

    s = 1.0
    for i in range(T_new - 1):
        s = -alpha * s * dt / tau_new + s

        addsum = 0.0
        for b in range(basis_num):
            Phi[b] = np.exp(-h[b] * (s - c[b])**2)
            addsum += Phi[b]

        newForce = np.matmul(Phi / addsum, W) * s

        obstacles_i = [p[:, i] for p in obs_paths]
        
        Ct = spatial_coupling_OA(DMP_pos[:, i], DMP_vel[:, i], obstacles_i,
                                 gamma=oa_gamma, beta=oa_beta) if len(obstacles_i) else 0.0 # checks if we have any obstacles to avoid

        DMP_acc[:, i] = (kp*(newGoal - DMP_pos[:, i]) - tau_new*kv*DMP_vel[:, i] + newForce + Ct) / (tau_new**2)

        DMP_pos[:, i+1] = DMP_pos[:, i] + DMP_vel[:, i] * dt
        DMP_vel[:, i+1] = DMP_vel[:, i] + DMP_acc[:, i] * dt

    return DMP_pos, DMP_vel, DMP_acc, W, tau_new, newStart, newGoal





# create a single obstacle path

def make_object_path(
    ref_path,            # (3, T_new) reference trajectory (e.g., demo pos resampled or DMP_pos)
    t,                   # (T_new,) time array matching rollout
    anchor_idx,          # int: where along ref_path to "place" the obstacle initially
    offset=(0.0, 0.0, 0.0), # (3,) constant offset from anchor point
    motion_fn=None,      # function: motion_fn(t_local, i_local, anchor_point, offset) -> (3,) delta
    spawn_idx=0,         # int: when obstacle "appears"
    keep_after_end=True, # if motion_fn stops early, keep last value
    eps=1e-12
):
    """
    Returns obj_path: (3, T_new)
    - Before spawn_idx: NaN (doesn't exist yet)
    - From spawn_idx onward: anchor_point + offset + motion_fn(...)
    
    motion_fn signature:
        delta = motion_fn(t_local, i_local, anchor_point, offset)
        where:
          t_local = t[i] - t[spawn_idx]   (time since spawn)
          i_local = i - spawn_idx         (step since spawn)
          anchor_point = ref_path[:, anchor_idx]
          offset = np.array(offset)
    If motion_fn is None -> static obstacle (delta = 0).
    """

    # Quick checks
    ref_path = np.asarray(ref_path, dtype=float)
    assert ref_path.ndim == 2 and ref_path.shape[0] == 3, "ref_path must be (3, T_new)"
    t = np.asarray(t, dtype=float).reshape(-1,) # t must be 1D array of length T_new
    T_new = ref_path.shape[1] # number of columns (length of trajectory in time steps)
    assert t.shape[0] == T_new, "t must have length T_new"
    assert 0 <= anchor_idx < T_new, "anchor_idx out of range"
    assert 0 <= spawn_idx < T_new, "spawn_idx out of range"


    offset = np.asarray(offset, dtype=float).reshape(3,) # convert to array and ensure shape (3,)
    anchor_point = ref_path[:, anchor_idx].copy() # coordinates of the anchor point on the reference path

    obj = np.full((3, T_new), np.nan, dtype=float) # initialize object path with NaN

    if motion_fn is None:
        # static object after spawn
        obj[:, spawn_idx:] = (anchor_point + offset).reshape(3, 1) #spawn_idx onward: anchor_point + offset
        return obj

    last_val = None
    t0 = t[spawn_idx]
    for i in range(spawn_idx, T_new): # from spawn_idx to end, we fill in the path according to motion_fn
        t_local = t[i] - t0 # time since spawn
        i_local = i - spawn_idx # step since spawn

        delta = motion_fn(t_local, i_local, anchor_point, offset) # get motion delta from a motion_fn
        delta = np.asarray(delta, dtype=float).reshape(3, )# ensure numpy array with shape (3,)

        val = anchor_point + offset + delta
        obj[:, i] = val # append to path  
        last_val = val

    if keep_after_end and last_val is not None:
        # (Already filled every step to end, so nothing needed.
        # This hook is here if you later decide motion_fn can return None to stop.)
        pass

    return obj



# spatial_coupling_OA function
def spatial_coupling_OA(y, ydot, obstacles, gamma=200.0, beta=10.0, eps=1e-9):
    """
    Spatial coupling obstacle avoidance (matches your notes).
    y:    (d,) position
    ydot: (d,) velocity
    obstacles: list/iterable of obstacle positions, each (d,)
    returns Ct: (d,) coupling term
    """

    # turn into 1D vectors
    y = np.asarray(y).reshape(-1,)
    ydot = np.asarray(ydot).reshape(-1,)

    d_dim = y.shape[0]
    Ct_total = np.zeros(d_dim)

    vnorm = np.linalg.norm(ydot) # magnitude of velocity
    if vnorm < eps:
        return Ct_total

    for o in obstacles:
        o = np.asarray(o).reshape(-1,)

        # ---- MINIMAL FIX #1: ignore "not spawned" obstacles (NaNs)
        ''' This is the key fix to handle moving obstacles that appear mid-rollout.'''
        if not np.all(np.isfinite(o)): # if any component of o is NaN, it means this obstacle hasn't spawned yet, so we skip this obstacle in the OA calculation
                                       # the other obstacles in the list that do exist will still contribute to OA
            continue

        dvec = o - y
        dnorm = np.linalg.norm(dvec) #magnitude of distance vector from current position to obstacle
        if dnorm < eps:
            continue

        cosang = np.dot(dvec, ydot) / (dnorm * vnorm + eps)
        cosang = np.clip(cosang, -1.0, 1.0)
        psi = np.arccos(cosang)

        if d_dim == 2:
            cross_z = dvec[0]*ydot[1] - dvec[1]*ydot[0]
            sign = 1.0 if cross_z >= 0 else -1.0
            Rydot = sign * np.array([-ydot[1], ydot[0]])

        elif d_dim == 3:
            r = np.cross(dvec, ydot) #
            rnorm = np.linalg.norm(r)

            if rnorm < eps:
                a = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(a, ydot) / (np.linalg.norm(a)*vnorm + eps)) > 0.9:
                    a = np.array([0.0, 1.0, 0.0])
                r = np.cross(a, ydot)
                rnorm = np.linalg.norm(r)
                if rnorm < eps:
                    continue

            rhat = r / (rnorm + eps)
            Rydot = np.cross(rhat, ydot) + rhat * np.dot(rhat, ydot)

        else:
            raise ValueError("Only 2D or 3D supported.")

        Ct_total += gamma * Rydot * psi * np.exp(-beta * psi)

    return Ct_total


# motion Functions

def motion_cross_x(t_local, i_local, anchor, offset, v=0.6):
    return np.array([v * t_local, 0.0, 0.0])

def motion_cross_y(t_local, i_local, anchor, offset, v=0.6):
    return np.array([0.0, v * t_local, 0.0])

def motion_drift_diag(t_local, i_local, anchor, offset, v=(0.0, 0.5, -0.1)):
    v = np.asarray(v, float)
    return v * t_local 

def motion_circle_xy(t_local, i_local, anchor, offset, R=0.22, w=2*np.pi*0.9):
    return np.array([R*np.cos(w*t_local) - R, R*np.sin(w*t_local), 0.0])