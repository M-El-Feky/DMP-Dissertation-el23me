import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data

from DMP_Experiment import draw_polyline


# SETTINGS
HOME_JOINT_ANGLES = [0.5, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8]

PROTECTED_LINK_NAME = "panda_hand"
PROTECTED_LOCAL_POINTS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.05]), # top
    np.array([0.02, 0.0,  0.05]), # front top
    np.array([-0.02, 0.0, 0.05]), # back top
    np.array([0.0, 0.22, 0.05]), # left top
    np.array([0.03, 0.22, 0.05]), # left front top
    np.array([-0.03, 0.22, 0.05]), # left back top
    np.array([0.03, -0.22, 0.05]), # right front top
    np.array([-0.03, -0.22, 0.05]), # right back top
    np.array([0.0, 0.0,-0.05]), # bottom
    np.array([0.02, 0.0, -0.05]), # front bottom
    np.array([-0.02, 0.0, -0.05]), # back bottom
    np.array([0.0,  0.22, -0.05]), # left bottom
    np.array([0.03, 0.22, -0.05]), # left front bottom
    np.array([-0.03, 0.22, -0.05]), # left back bottom
    np.array([0.03, -0.22, -0.05]), # right front bottom
    np.array([-0.03, -0.22, -0.05]) # right back bottom
]
KP=150
KV=25

OA_ETA = 0.05 # strength of the OA repulsive force
OA_D0 = 0.12 # distance at which OA starts to have an effect 0.15
OA_LAMBDA = 0.1 # scaling of the OA coupling term
GOAL_PULL = 0.01


# Joint-space OA function
def joint_space_OA(
    q, obstacle_position,
    robot, movable_indices, link_name_to_idx,
    protected_link_name, protected_local_points,
    oa_eta=0.05, oa_d0=0.12, oa_lambda=0.2
):
    point_list, jacobian_list = get_protected_points_and_jacobians(
        q,
        robot,
        movable_indices,
        link_name_to_idx,
        protected_link_name,
        protected_local_points
    )

    obstacle_position = np.asarray(obstacle_position).reshape(3,)
    Ct_total = np.zeros(7)

    for x, J in zip(point_list, jacobian_list):
        d = x - obstacle_position
        r = np.linalg.norm(d)

        if r < 1e-9 or r > oa_d0:
            continue

        F_rep = oa_eta * (1.0 / r - 1.0 / oa_d0) * d / (r**3 + 1e-9)
        Ct_total += oa_lambda * (J.T @ F_rep)

    return Ct_total

# DMP Function
def dmp_discrete_7d_OA(
    pos, dt,
    robot,
    movable_indices,
    link_name_to_idx,
    protected_link_name, 
    protected_local_points,
    kp=150.0,
    kv=25.0, 
    alpha=5.0, 
    basis_num=40,
    speed=1.0,
    start_shift=None,
    goal_shift=None,
    spatial_scale=None,
    init_vel_scale=1.0,
    init_vel_offset=None,
    obstacle_position=None,
    oa_eta=0.05,
    oa_d0=0.12,
    oa_lambda=0.2,
    goal_pull=0.15
):   
    
    
    pos = np.asarray(pos)
    assert pos.ndim == 2 and pos.shape[0] == 7, "pos must be shape (7, T)"
    assert speed > 0, "speed must be > 0"

    T = pos.shape[1]

    tau_demo = dt * T 
    tau_new  = tau_demo / speed
    T_new = int(np.round(tau_new / dt))
    T_new = max(T_new, 2)

    vel = np.gradient(pos, axis=-1)/dt 
    acc = np.gradient(vel, axis=-1)/dt 

    goal = pos[:, -1]
    demo_start = pos[:, 0]
    demo_goal  = pos[:, -1]

    if start_shift is None:
        start_shift = np.zeros(7)
    if goal_shift is None:
        goal_shift = np.zeros(7)

    start_shift = np.asarray(start_shift).reshape(7,)
    goal_shift  = np.asarray(goal_shift).reshape(7,)

    newStart = demo_start + start_shift
    newGoal  = demo_goal  + goal_shift

    if init_vel_offset is None:
        init_vel_offset = np.zeros(7)
    init_vel_offset = np.asarray(init_vel_offset).reshape(7,)

    # spatial scaling factor scales the forcing term so the trajectory shape stretches or shrinks
    demo_displacement = demo_goal - demo_start
    new_displacement  = newGoal - newStart

    if spatial_scale is None:
        spatial_scale = np.ones(7)
        for d in range(7):
            if abs(demo_displacement[d]) > 1e-8:
                spatial_scale[d] = new_displacement[d] / demo_displacement[d]
    else:
        spatial_scale = np.asarray(spatial_scale).reshape(7,)

    # goal gate fades the OA coupling term near the goal
    initial_goal_error = np.linalg.norm(newGoal - newStart) + 1e-9
    goal_radius = goal_pull * initial_goal_error

    # basis setup
    u = np.zeros(basis_num)
    c = np.zeros(basis_num)
    h = np.zeros(basis_num)

    # control center loactions of gaussians
    for i in range(basis_num):
        u[i] = (1.0 / basis_num) * i
        c[i] = np.exp(-alpha * u[i])

    # control spread of gaussians
    for i in range(basis_num - 1):
        h[i] = 0.5 / ((0.65 * (c[i+1] - c[i]))**2) # controling 
    h[basis_num - 1] = h[basis_num - 2]

    #  Learning
    Phi = np.zeros(basis_num)
    Phi_total = np.zeros((T, basis_num))
    force = np.zeros((T, 7))

    s = 1.0
    for i in range(T):
        # canonical system 
        s = -alpha * s * dt / tau_demo  + s

        addsum = 0.0
        for b in range(basis_num):
            Phi[b] = np.exp(-h[b] * (s - c[b])**2)
            addsum += Phi[b]
        Phi = Phi / addsum * s
        Phi_total[i, :] = Phi

        # target forcing term
        force[i, :] = (tau_demo*tau_demo)*acc[:, i] - kp*(goal - pos[:, i]) + tau_demo*kv*vel[:, i]

    trainPattern = np.matmul(np.linalg.pinv(Phi_total), force)  # (basis_num, 7)

    # setup for reconstruction
    DMP_pos = np.zeros((7, T_new))
    DMP_vel = np.zeros((7, T_new))
    DMP_acc = np.zeros((7, T_new))

    DMP_pos[:, 0] = newStart 
    DMP_vel[:, 0] = init_vel_scale * vel[:, 0] + init_vel_offset

    s = 1.0
    for i in range(T_new - 1):
        s = -alpha * s * dt / tau_new + s
        addsum = 0.0
        
        for b in range(basis_num):
            Phi[b] = np.exp(-h[b] * (s - c[b])**2)
            addsum = addsum + Phi[b]

        newForce = np.matmul(Phi / addsum, trainPattern) * s * spatial_scale

        if obstacle_position is None:
            Ct = np.zeros(7)
        else:
            # obstacle avoidance coupling term
            Ct = joint_space_OA(
                DMP_pos[:, i],
                obstacle_position,
                robot,
                movable_indices,
                link_name_to_idx,
                protected_link_name,
                protected_local_points,
                oa_eta=oa_eta,
                oa_d0=oa_d0,
                oa_lambda=oa_lambda
            )

            # fade OA near the goal
            goal_error = np.linalg.norm(newGoal - DMP_pos[:, i])
            if goal_error <= goal_radius:
                Ct = Ct * (goal_error / goal_radius)

        # transformation system
        DMP_acc[:, i] = (kp*(newGoal - DMP_pos[:, i]) - tau_new*kv*DMP_vel[:, i] + newForce + Ct) / (tau_new**2)
        DMP_pos[:, i+1] = DMP_pos[:, i] + DMP_vel[:, i] * dt
        DMP_vel[:, i+1] = DMP_vel[:, i] + DMP_acc[:, i] * dt

    return DMP_pos, DMP_vel, DMP_acc, trainPattern, tau_new, newStart, newGoal


# Demonstration Creation
def make_demo_path(T=700):
    u = np.linspace(0, 1.2, T)

    x1 = HOME_JOINT_ANGLES[0] + 0.45 * u
    x2 = HOME_JOINT_ANGLES[1] + 0.30 * np.sin(np.pi * u)
    x3 = HOME_JOINT_ANGLES[2] + 0.35 * np.sin(np.pi * u)
    x4 = HOME_JOINT_ANGLES[3] + 0.55 * u
    x5 = HOME_JOINT_ANGLES[4] + 0.15 * np.sin(2 * np.pi * u)
    x6 = HOME_JOINT_ANGLES[5] - 0.25 * np.sin(np.pi * u)
    x7 = HOME_JOINT_ANGLES[6] - 0.28 * u

    return np.vstack([x1, x2, x3, x4, x5, x6, x7])


# Pybullet Setup
def setup():
    p.connect(p.GUI) # connect to PyBullet with GUI, change to p.DIRECT for headless
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation() # reset to default state (removes any previous objects ect)
    p.setGravity(0, 0, -9.81) # set gravity
    p.setRealTimeSimulation(0)

    p.loadURDF("plane.urdf") # load a plane
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True) # load the Franka Panda
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # improve vissulization

    # Find the end-effector link index
    ee = None
    movable_indices = []
    link_name_to_idx = {}

    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        q_index = info[3]
        link_name = info[12].decode("utf-8", errors="replace")

        link_name_to_idx[link_name] = j

        if link_name == "panda_hand":
            ee = j

        if q_index > -1:
            movable_indices.append(j)

    if ee is None:
        raise RuntimeError("could not find panda_hand")

    return robot, ee, movable_indices, link_name_to_idx


def set_home(robot):
    for j in range(7):
        p.resetJointState(robot, j, HOME_JOINT_ANGLES[j])


def set_arm_joints(robot, q):
    for j in range(7):
        p.resetJointState(robot, j, float(q[j]))


def get_link_pose(robot, link_idx):
    st = p.getLinkState(
        robot,
        link_idx,
        computeForwardKinematics=True
    )

    pos = np.array(st[4], dtype=float)
    R = np.array(p.getMatrixFromQuaternion(st[5]), dtype=float).reshape(3, 3)

    return pos, R


def get_hand_position(robot, ee):
    pos, _ = get_link_pose(robot, ee)
    return pos


def get_protected_points_and_jacobians(
    q, robot, movable_indices, link_name_to_idx,
    protected_link_name, protected_local_points
):
    set_arm_joints(robot, q)
    link_idx = link_name_to_idx[protected_link_name]

    pos, R = get_link_pose(robot, link_idx) # get the position and orientation of the protected link

    n_movable = len(movable_indices)
    q_full = [0.0] * n_movable
    for j in range(7):
        q_full[j] = float(q[j])

    zero_vec = [0.0] * n_movable

    point_list = []
    jacobian_list = []
    # get the world position and jacobian for each protected local point on the robot
    for local_point in protected_local_points:
        x = pos + R @ local_point

        J_lin, _ = p.calculateJacobian( # get the jacobian for the protected point
            robot,
            link_idx,
            list(map(float, local_point)),
            q_full,
            zero_vec,
            zero_vec
        )

        J = np.asarray(J_lin, dtype=float)[:, :7] # only take linear part and first 7 columns for the arm joints

        point_list.append(x)
        jacobian_list.append(J)

    return point_list, jacobian_list


def joint_rollout_to_hand_path(joint_rollout, robot, ee):
    T = joint_rollout.shape[1]
    path = np.zeros((T, 3))

    for k in range(T):
        set_arm_joints(robot, joint_rollout[:, k])
        path[k] = get_hand_position(robot, ee)

    return path


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


def min_distance_to_obstacle(path, obstacle):
    obstacle = np.asarray(obstacle).reshape(1, 3)
    return np.min(np.linalg.norm(path - obstacle, axis=1))


# Plotting results
def plot_oa_results(demo_pos, dmp_nom, dmp_oa, obstacle_position, dt):
    demo_xyz = demo_pos
    dmp_nom_xyz = dmp_nom
    dmp_oa_xyz = dmp_oa

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(demo_xyz[:, 0], demo_xyz[:, 1], demo_xyz[:, 2], linewidth=2, label="Demo")
    ax.plot(dmp_nom_xyz[:, 0], dmp_nom_xyz[:, 1], dmp_nom_xyz[:, 2], "--", linewidth=2, label="Normal DMP")
    ax.plot(dmp_oa_xyz[:, 0], dmp_oa_xyz[:, 1], dmp_oa_xyz[:, 2], linewidth=2, label="OA DMP")

    ax.scatter(obstacle_position[0], obstacle_position[1], obstacle_position[2], s=60, label="Obstacle")
    ax.scatter(demo_xyz[0, 0], demo_xyz[0, 1], demo_xyz[0, 2], s=50, label="Start")
    ax.scatter(demo_xyz[-1, 0], demo_xyz[-1, 1], demo_xyz[-1, 2], s=50, label="Goal")

    ax.set_title("7D DMP Obstacle Avoidance")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

    err_nom, _, _ = compute_errors(demo_pos, dmp_nom)
    err_oa, _, _ = compute_errors(demo_pos, dmp_oa)

    t_axis = np.arange(len(err_nom)) * dt
    plt.figure(figsize=(9, 4))
    plt.plot(t_axis, err_nom, linewidth=2, label="Demo vs normal DMP")
    plt.plot(t_axis, err_oa, linewidth=2, label="Demo vs OA DMP")
    plt.title("Tracking Error Through Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def stable_pose_ik(robot, ee, p_target, q_target, previous_state):
    ll = [-2.9, -1.8, -2.9, -3.1, -2.9, -0.1, -2.9]
    ul = [ 2.9,  1.8,  2.9,  0.0,  2.9,  3.7,  2.9]
    jr = [u - l for l, u in zip(ll, ul)]

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


def draw_valid_polyline(points, rgb=(0, 0, 1), width=2):
    points = np.asarray(points, dtype=float)
    valid = np.all(np.isfinite(points), axis=1)
    pts = points[valid]
    if len(pts) >= 2:
        draw_polyline(pts, rgb=rgb, width=width)


def run_robot_tool_tracking(tool_des, dt, o_local, target_orn, obs_paths=None, sleep=True):
    robot, ee, movable_indices, link_name_to_idx = setup()
    arm_joints = list(range(7))
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

    R_d = np.array(p.getMatrixFromQuaternion(target_orn), dtype=float).reshape(3, 3)
    previous_state = [p.getJointState(robot, j)[0] for j in arm_joints]

    tool_exec_log = []
    hand_exec_log = []
    hand_target_log = []

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

        st = p.getLinkState(robot, ee, computeForwardKinematics=True)
        p_hand = np.array(st[4], dtype=float)
        R_hand = np.array(p.getMatrixFromQuaternion(st[5]), dtype=float).reshape(3, 3)
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


# Main experiment function
def main():
    t = 10.0
    dt = 0.01
    T = int(np.round(t / dt))

    # demonstration
    pos = make_demo_path(T=T)   # (7, T)

    # pybullet model
    robot, ee, movable_indices, link_name_to_idx = setup()
    set_home(robot)

    hand_demo = joint_rollout_to_hand_path( # hand path in cartesian space
        pos,
        robot,
        ee
    )

    # normal DMP
    DMP_nom, _, _, _, _, _, _ = dmp_discrete_7d_OA(
        pos,
        dt,
        robot,
        movable_indices,
        link_name_to_idx,
        PROTECTED_LINK_NAME,
        PROTECTED_LOCAL_POINTS,
        speed=1.0,
        start_shift=np.zeros(7),
        goal_shift=np.zeros(7),
        spatial_scale=None,
        init_vel_scale=1.0,
        init_vel_offset=np.zeros(7),
        obstacle_position=None
    )

    # one obstacle near the hand path
    anchor_idx = int(0.5 * T)
    obstacle_position = hand_demo[anchor_idx] + np.array([0.03, 0.03, 0.04])

    # OA DMP
    DMP_oa, _, _, _, _, _, _ = dmp_discrete_7d_OA(
        pos,
        dt,
        robot,
        movable_indices,
        link_name_to_idx,
        PROTECTED_LINK_NAME,
        PROTECTED_LOCAL_POINTS,
        kp=KP,
        kv=KV,
        alpha=5.0,
        basis_num=40,
        speed=1.0,
        start_shift=np.zeros(7),
        goal_shift=np.zeros(7),
        spatial_scale=None,
        init_vel_scale=1.0,
        init_vel_offset=np.zeros(7),
        obstacle_position=obstacle_position,
        oa_eta=OA_ETA,
        oa_d0=OA_D0,
        oa_lambda=OA_LAMBDA,
        goal_pull=GOAL_PULL
    )

    # hand paths for both rollouts
    hand_nom = joint_rollout_to_hand_path(
        DMP_nom,
        robot,
        ee
    )

    hand_oa = joint_rollout_to_hand_path(
        DMP_oa,
        robot,
        ee
    )

    p.disconnect()

    target_orn = p.getQuaternionFromEuler([3.1416, 0.0, 0.0])
    o_local = np.zeros(3)

    # make a static obstacle path so the execution function can display it
    obs_static = np.tile(obstacle_position.reshape(1, 3), (len(hand_oa), 1))

    tool_exec_log, hand_exec_log, hand_target_log = run_robot_tool_tracking(
        tool_des=hand_oa,
        dt=dt,
        o_local=o_local,
        target_orn=target_orn,
        obs_paths=[obs_static],
        sleep=True
    )


    # plot results
    plot_oa_results(
        demo_pos=hand_demo,
        dmp_nom=hand_nom,
        dmp_oa=hand_oa,
        obstacle_position=obstacle_position,
        dt=dt
    )


    # metrics
    print("Normal DMP results")
    print(f"Path length:{path_length(hand_nom):.6f}")
    print(f"Final error:{compute_errors(hand_demo, hand_nom)[2]:.6f}")
    print(f"RMS error:{compute_errors(hand_demo, hand_nom)[1]:.6f}")
    print(f"Min dist obstacle:{min_distance_to_obstacle(hand_nom, obstacle_position):.6f}")

    print("\nOA DMP results")
    print(f"Path length:{path_length(hand_oa):.6f}")
    print(f"Final error:{compute_errors(hand_demo, hand_oa)[2]:.6f}")
    print(f"RMS error:{compute_errors(hand_demo, hand_oa)[1]:.6f}")
    print(f"Min dist obstacle:{min_distance_to_obstacle(hand_oa, obstacle_position):.6f}")

    p.disconnect()

if __name__ == "__main__":
    main()