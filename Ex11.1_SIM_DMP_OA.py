import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data

from DMP_OA_Functions import (
    dmp_discrete_3d_moving_OA,
    make_object_path,
    motion_cross_y,
)

# DMP Function
def dmp_discrete_3d_reproduce(
    pos, 
    dt,
    kp=150.0, 
    kv=25, 
    alpha=5.0, 
    basis_num=40,
    speed=1.0, 
    start_shift=None, 
    goal_shift=None,
    init_vel_scale=1.0, 
    init_vel_offset=None
):
    pos = np.asarray(pos)
    assert pos.ndim == 2 and pos.shape[0] == 3, "pos must be shape (3, T)"
    assert speed > 0, "speed must be > 0"

    T = pos.shape[1]

    tau_demo = dt * T
    tau_new = tau_demo / speed

    vel = np.gradient(pos, axis=-1)/dt
    acc = np.gradient(vel, axis=-1)/dt

    goal = pos[:, -1]
    demo_start = pos[:, 0]
    demo_goal = pos[:, -1]

    if start_shift is None:
        start_shift = np.zeros(3)
    if goal_shift is None:
        goal_shift = np.zeros(3)

    start_shift = np.asarray(start_shift).reshape(3,)
    goal_shift = np.asarray(goal_shift).reshape(3,)

    newStart = demo_start + start_shift
    newGoal = demo_goal + goal_shift

    if init_vel_offset is None:
        init_vel_offset = np.zeros(3)
    init_vel_offset = np.asarray(init_vel_offset).reshape(3,)

    # Basis setup
    u = np.zeros(basis_num)
    c = np.zeros(basis_num)
    h = np.zeros(basis_num)

    for i in range(basis_num):
        u[i] = (1.0 / basis_num) * i
        c[i] = np.exp(-alpha * u[i])

    for i in range(basis_num - 1):
        h[i] = 0.5 / ((0.65 * (c[i + 1] - c[i])) ** 2)
    h[basis_num - 1] = h[basis_num - 2]

    # Learning
    Phi = np.zeros(basis_num)
    Phi_total = np.zeros((T, basis_num))
    force = np.zeros((T, 3))

    s = 1.0
    for i in range(T):
        s = -alpha * s * dt / tau_demo + s

        addsum = 0.0
        for b in range(basis_num):
            Phi[b] = np.exp(-h[b] * (s - c[b]) ** 2)
            addsum += Phi[b]

        Phi = Phi / addsum * s
        Phi_total[i, :] = Phi

        force[i, :] = (tau_demo*tau_demo)*acc[:, i] - kp*(goal - pos[:, i]) + tau_demo*kv*vel[:, i]


    trainPattern = np.matmul(np.linalg.pinv(Phi_total), force)

    # Reconstruction
    T_new = int(np.round(tau_new / dt))

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
            Phi[b] = np.exp(-h[b] * (s - c[b]) ** 2)
            addsum += Phi[b]

        newForce = np.matmul(Phi / addsum, trainPattern) * s

        DMP_acc[:, i] = (kp*(newGoal - DMP_pos[:, i]) - tau_new*kv*DMP_vel[:, i] + newForce) / (tau_new**2)

        DMP_pos[:, i + 1] = DMP_pos[:, i] + DMP_vel[:, i] * dt
        DMP_vel[:, i + 1] = DMP_vel[:, i] + DMP_acc[:, i] * dt

    return DMP_pos, DMP_vel, DMP_acc, trainPattern, tau_new, newStart, newGoal


def draw_valid_polyline(points, rgb=(0, 0, 1), width=2):
    points = np.asarray(points, dtype=float)
    valid = np.all(np.isfinite(points), axis=1)
    pts = points[valid]
    if len(pts) >= 2:
        draw_polyline(pts, rgb=rgb, width=width)


def min_distance_to_obstacles(path_xyz, obs_paths_xyz):
    dmin = np.inf
    for obs in obs_paths_xyz:
        valid = np.all(np.isfinite(obs), axis=1)
        obs_valid = obs[valid]
        if len(obs_valid) == 0:
            continue
        for p_i in path_xyz:
            d = np.linalg.norm(obs_valid - p_i, axis=1)
            dmin = min(dmin, np.min(d))
    return dmin



# Demonstration Creation
def make_demo_path(T=500):
    u = np.linspace(0, 1, T)

    x = 0.35 + 0.25 * u
    y = -0.15 + 0.30 * u + 0.05 * np.sin(np.pi * u)
    z = 0.35 + 0.10 * np.sin(np.pi * u)

    return np.vstack([x, y, z])


# Pybullet Setup
def setup():
    p.connect(p.GUI) # connect to PyBullet with GUI, change to p.DIRECT for headless
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation() # reset to default state (removes any previous objects ect)
    p.setGravity(0, 0, -9.81) # set gravity
    p.setRealTimeSimulation(0)

    p.loadURDF("plane.urdf") # load a plane
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)# load the Franka Panda
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # improve vissulization

    # Find the end-effector link index
    ee = None
    for j in range(p.getNumJoints(robot)):
        link_name = p.getJointInfo(robot, j)[12].decode("utf-8", errors="replace")
        if link_name == "panda_hand":
            ee = j
            break

    if ee is None:
        raise RuntimeError("could not find panda_hand")

    arm_joints = list(range(7))
    return robot, ee, arm_joints


def set_home(robot):
    home_q = [0.5, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8]
    for j in range(7):
        p.resetJointState(robot, j, home_q[j])


def quat_to_R(quat): # convert quaternion to rotation matrix
    return np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)


def stable_pose_ik(robot, ee, p_target, q_target, previous_state):
    ll = [-2.9, -1.8, -2.9, -3.1, -2.9, -0.1, -2.9] # lower joint limits
    ul = [ 2.9,  1.8,  2.9,  0.0,  2.9,  3.7,  2.9] # upper joint limits
    jr = [u - l for l, u in zip(ll, ul)] # joint ranges

    q_full = p.calculateInverseKinematics(
        robot,
        ee,
        targetPosition=p_target,
        targetOrientation=q_target,
        lowerLimits=ll,
        upperLimits=ul,
        jointRanges=jr,
        restPoses=previous_state,
        maxNumIterations=150,
        residualThreshold=1e-6,
    )
    return list(q_full[:7])


def draw_polyline(points, rgb=(0, 0, 0), width=2):
    for i in range(len(points) - 1):
        p.addUserDebugLine(
            points[i].tolist(),
            points[i + 1].tolist(),
            lineColorRGB=list(rgb),
            lineWidth=width,
            lifeTime=0
        )


# Robot execution of tool tracking

def run_robot_tool_tracking(tool_des, dt, o_local, target_orn, obs_paths=None, sleep=True):
    robot, ee, arm_joints = setup()
    set_home(robot)

    if obs_paths is None:
        obs_paths = []

    view_center = np.mean(tool_des, axis=0)
    p.resetDebugVisualizerCamera(1.2, 55, -25, view_center.tolist())

    substeps = 3
    p.setTimeStep(dt / substeps)

    p.removeAllUserDebugItems()
    draw_polyline(tool_des, rgb=(0, 0, 0), width=2)

    for obs in obs_paths:
        draw_valid_polyline(obs, rgb=(0, 0, 1), width=2)

    # create spheres to visualize obstacles in the simulation
    obs_ids = []
    for _ in obs_paths:
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[0.1, 0, 1.0, 0.9]
        )
        obs_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=vis,
            baseCollisionShapeIndex=-1,
            basePosition=[0, 0, -5]
        )
        obs_ids.append(obs_id)

    R_d = quat_to_R(target_orn)
    previous_state = [p.getJointState(robot, j)[0] for j in arm_joints]

    tool_exec_log = [] # record the executed tool positions
    hand_exec_log = [] # record the executed hand positions
    hand_target_log = [] # record the target hand positions 

    p_tool_start = tool_des[0]
    p_hand_start = (p_tool_start - R_d @ o_local).tolist()

    q_start = stable_pose_ik(robot, ee, p_hand_start, target_orn, previous_state)
    previous_state = q_start

    settle_time = 1.0
    settle_steps = int(np.round(settle_time / (dt / substeps)))
    for _ in range(settle_steps):
        p.setJointMotorControlArray(
            robot,
            arm_joints,
            p.POSITION_CONTROL,
            targetPositions=q_start,
            forces=[280] * 7
        )
        p.stepSimulation()
        if sleep:
            time.sleep(dt / substeps)

    for k in range(len(tool_des)):
        # update obstacle visuals
        for obs_id, obs in zip(obs_ids, obs_paths):
            if k < len(obs) and np.all(np.isfinite(obs[k])):
                p.resetBasePositionAndOrientation(obs_id, obs[k].tolist(), [0, 0, 0, 1])
            else:
                p.resetBasePositionAndOrientation(obs_id, [0, 0, -5], [0, 0, 0, 1])

        p_tool_target = tool_des[k]
        p_hand_target = (p_tool_target - R_d @ o_local).tolist()

        q_arm = stable_pose_ik(robot, ee, p_hand_target, target_orn, previous_state)
        previous_state = q_arm

        for _ in range(substeps):
            p.setJointMotorControlArray(
                robot,
                arm_joints,
                p.POSITION_CONTROL,
                targetPositions=q_arm,
                forces=[280] * 7
            )
            p.stepSimulation()
            if sleep:
                time.sleep(dt / substeps)

        st = p.getLinkState(robot, ee)
        p_hand = np.array(st[4], dtype=float)
        R_hand = quat_to_R(st[5])
        p_tool_exec = p_hand + R_hand @ o_local

        tool_exec_log.append(p_tool_exec.copy())
        hand_exec_log.append(p_hand.copy())
        hand_target_log.append(np.array(p_hand_target, dtype=float))

    for _ in range(10):
        p.setJointMotorControlArray(
            robot,
            arm_joints,
            p.POSITION_CONTROL,
            targetPositions=q_arm,
            forces=[280] * 7
        )
        p.stepSimulation()
        if sleep:
            time.sleep(dt / substeps)

    tool_exec_log = np.array(tool_exec_log)
    hand_exec_log = np.array(hand_exec_log)
    hand_target_log = np.array(hand_target_log)

    draw_polyline(tool_exec_log, rgb=(1, 0, 0), width=2)

    return tool_exec_log, hand_exec_log, hand_target_log

# metrics computation
def path_length(path):
    diffs = np.diff(path, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


def compute_errors(ref_path, test_path):
    N = min(len(ref_path), len(test_path))
    ref_cut = ref_path[:N]
    test_cut = test_path[:N]

    point_error = np.linalg.norm(test_cut - ref_cut, axis=1)
    rms_error = np.sqrt(np.mean(point_error ** 2))
    final_error = np.linalg.norm(test_cut[-1] - ref_cut[-1])

    return point_error, rms_error, final_error


# Plotting results
def plot_oa_results(demo_pos, dmp_nom, dmp_oa, tool_exec_oa, obs_paths_xyz, dt):
    demo_xyz = demo_pos.T
    dmp_nom_xyz = dmp_nom.T
    dmp_oa_xyz = dmp_oa.T

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(demo_xyz[:, 0], demo_xyz[:, 1], demo_xyz[:, 2], linewidth=2, label="Demo")
    ax.plot(dmp_nom_xyz[:, 0], dmp_nom_xyz[:, 1], dmp_nom_xyz[:, 2], "--", linewidth=2, label="Normal DMP")
    ax.plot(dmp_oa_xyz[:, 0], dmp_oa_xyz[:, 1], dmp_oa_xyz[:, 2], linewidth=2, label="OA DMP")
    ax.plot(tool_exec_oa[:, 0], tool_exec_oa[:, 1], tool_exec_oa[:, 2], linewidth=2, label="Robot executed OA tool path")

    for i, obs in enumerate(obs_paths_xyz):
        valid = np.all(np.isfinite(obs), axis=1)
        ax.plot(obs[valid, 0], obs[valid, 1], obs[valid, 2], linewidth=2, label=f"Obstacle {i+1}")

    ax.scatter(demo_xyz[0, 0], demo_xyz[0, 1], demo_xyz[0, 2], s=50, label="Start")
    ax.scatter(demo_xyz[-1, 0], demo_xyz[-1, 1], demo_xyz[-1, 2], s=50, label="Goal")

    ax.set_title("3D DMP Obstacle Avoidance in PyBullet")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("OA Experiment results")
    print(f"Normal DMP path length:{path_length(dmp_nom_xyz):.6f}")
    print(f"OA DMP path length:{path_length(dmp_oa_xyz):.6f}")
    print(f"Executed OA tool path length:{path_length(tool_exec_oa):.6f}")
    print(f"Min dist normal DMP -> obs:{min_distance_to_obstacles(dmp_nom_xyz, obs_paths_xyz):.6f}")
    print(f"Min dist OA DMP -> obs:{min_distance_to_obstacles(dmp_oa_xyz, obs_paths_xyz):.6f}")
    print(f"Min dist exec tool -> obs:{min_distance_to_obstacles(tool_exec_oa, obs_paths_xyz):.6f}")


# Main experiment function
def main():
    t = 5.0
    dt = 0.01
    T = int(np.round(t / dt))

    o_local = np.array([0.00, 0.00, 0.10])
    target_orn = p.getQuaternionFromEuler([3.1416, 0.0, 0.0])

    # demonstration
    pos = make_demo_path(T=T)   # (3, T)
    t_arr = np.arange(T) * dt

    DMP_nom, _, _, _, _, _, _ = dmp_discrete_3d_reproduce(
        pos,
        dt,
        speed=1.0,
        start_shift=np.zeros(3),
        goal_shift=np.zeros(3),
        init_vel_scale=1.0,
        init_vel_offset=np.zeros(3),
    )


    # nominal DMP
    tool_exec_nom, hand_exec_nom, hand_target_nom = run_robot_tool_tracking(
        tool_des=DMP_nom.T,
        dt=dt,
        o_local=o_local,
        target_orn=target_orn,
        obs_paths=None,
        sleep=True,
    )
    p.disconnect()

    # one moving obstacle
    anchor_idx = int(0.55 * T)
    spawn_idx = int(0.20 * T)

    obs1 = make_object_path(
        ref_path=pos,
        t=t_arr,
        anchor_idx=anchor_idx,
        offset=(0.0, 0.1, 0.0),
        motion_fn=lambda t_local, i_local, anchor, offset: motion_cross_y(t_local, i_local, anchor, offset, v=-0.12),
        spawn_idx=spawn_idx,
        keep_after_end=True
    )

    obs_paths = [obs1]# each one is (3, T)

    # OA DMP
    DMP_oa, DMP_oa_vel, DMP_oa_acc, W_oa, tau_new, newStart, newGoal = dmp_discrete_3d_moving_OA(
        pos,
        dt,
        obs_paths=obs_paths,
        kp=150.0,
        kv=25.0,
        alpha=5.0,
        basis_num=40,
        speed=1.0,
        start_shift=np.zeros(3),
        goal_shift=np.zeros(3),
        init_vel_scale=1.0,
        init_vel_offset=np.zeros(3),
        oa_gamma=1000,
        oa_beta=6.0
    )

    # send OA DMP to robot
    tool_des_oa = DMP_oa.T # robot tracking uses (T, 3)
    obs_paths_sim = [obs.T for obs in obs_paths] # convert obstacles to (T, 3)

    tool_exec_log, hand_exec_log, hand_target_log = run_robot_tool_tracking(
        tool_des=tool_des_oa,
        dt=dt,
        o_local=o_local,
        target_orn=target_orn,
        obs_paths=obs_paths_sim,
        sleep=True
    )

    # plot results
    plot_oa_results(
        demo_pos=pos,
        dmp_nom=DMP_nom,
        dmp_oa=DMP_oa,
        tool_exec_oa=tool_exec_log,
        obs_paths_xyz=obs_paths_sim,
        dt=dt
    )
    p.disconnect()


if __name__ == "__main__":
    main()