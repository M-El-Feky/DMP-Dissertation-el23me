import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data


# ==================================================
# DMP FUNCTION
# ==================================================
def dmp_discrete_3d_reproduce(
    pos, dt,
    kp=150.0, kv=25, alpha=5.0, basis_num=40,
    speed=1.0, start_shift=None, goal_shift=None,
    init_vel_scale=1.0, init_vel_offset=None
):
    pos = np.asarray(pos)
    assert pos.ndim == 2 and pos.shape[0] == 3, "pos must be shape (3, T)"
    assert speed > 0, "speed must be > 0"

    T = pos.shape[1]

    tau_demo = dt * T
    tau_new = tau_demo / speed

    vel = np.gradient(pos, axis=-1) / dt
    acc = np.gradient(vel, axis=-1) / dt

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


# ==================================================
# DEMONSTRATION PATH
# ==================================================
def make_demo_path(T=500):
    u = np.linspace(0, 1, T)

    x = 0.35 + 0.25 * u
    y = -0.15 + 0.30 * u + 0.05 * np.sin(np.pi * u)
    z = 0.35 + 0.10 * np.sin(np.pi * u)

    return np.vstack([x, y, z])


# ==================================================
# PYBULLET HELPERS
# ==================================================
def setup():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    ee = None
    for j in range(p.getNumJoints(robot)):
        link_name = p.getJointInfo(robot, j)[12].decode("utf-8", errors="replace")
        if link_name == "panda_hand":
            ee = j
            break

    if ee is None:
        raise RuntimeError("Could not find panda_hand")

    arm_joints = list(range(7))
    return robot, ee, arm_joints


def set_home(robot):
    home_q = [0.5, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8]
    for j in range(7):
        p.resetJointState(robot, j, home_q[j])


def quat_to_R(quat):
    return np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)


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


def draw_polyline(points, rgb=(0, 0, 0), width=2):
    for i in range(len(points) - 1):
        p.addUserDebugLine(
            points[i].tolist(),
            points[i + 1].tolist(),
            lineColorRGB=list(rgb),
            lineWidth=width,
            lifeTime=0
        )


# ==================================================
# ROBOT EXECUTION
# ==================================================
def run_robot_tool_tracking(tool_des, dt, o_local, target_orn, sleep=True):
    robot, ee, arm_joints = setup()
    set_home(robot)

    view_center = np.mean(tool_des, axis=0)
    p.resetDebugVisualizerCamera(1.2, 55, -25, view_center.tolist())
    substeps = 3 # number of simulation steps to take between each target point, allows for smoother execution

    p.setTimeStep(dt/substeps) # set the physics time step to be smaller than the control time step for smoother simulation

    p.removeAllUserDebugItems()
    draw_polyline(tool_des, rgb=(0, 0, 0), width=2)

    R_d = quat_to_R(target_orn)
    previous_state = [p.getJointState(robot, j)[0] for j in arm_joints]

    tool_exec_log = []
    hand_exec_log = []
    hand_target_log = []



    p_tool_start = tool_des[0]
    p_hand_start = (p_tool_start - R_d @ o_local).tolist()

    q_start = stable_pose_ik(robot, ee, p_hand_start, target_orn, previous_state)
    previous_state = q_start

    # Hold this pose briefly so the robot settles before tracking begins
    #settle_steps = int(np.round(1.0 / dt))   # 1 second

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
        p_tool_target = tool_des[k]
        p_hand_target = (p_tool_target - R_d @ o_local).tolist()

        q_arm = stable_pose_ik(robot, ee, p_hand_target, target_orn, previous_state)
        previous_state = q_arm

        for _ in range(substeps): # allows to step simulation several times and update the robot's position more smoothly between target points
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
        hand_target_log.append(p_hand_target.copy())
    for _ in range(10): # allows to step simulation several times and update the robot's position more smoothly between target points
        p.setJointMotorControlArray(
            robot,
            arm_joints,
            p.POSITION_CONTROL,
            targetPositions=q_arm,
            forces=[280] * 7
        )
        p.stepSimulation()
        time.sleep(dt/substeps) 

    tool_exec_log = np.array(tool_exec_log)
    hand_exec_log = np.array(hand_exec_log)
    hand_target_log = np.array(hand_target_log)

    draw_polyline(tool_exec_log, rgb=(1, 0, 0), width=2)

    return tool_exec_log, hand_exec_log, hand_target_log

# ==================================================
# METRICS
# ==================================================
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


# ==================================================
# PLOTTING
# ==================================================
def plot_all_results(demo_pos, dmp_pos, tool_exec, hand_exec, hand_target, dt):
    demo_xyz = demo_pos.T
    dmp_xyz = dmp_pos.T
    exec_xyz = tool_exec

    # 3D paths
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(demo_xyz[:, 0], demo_xyz[:, 1], demo_xyz[:, 2], linewidth=2, label="Demonstration")
    ax.plot(dmp_xyz[:, 0], dmp_xyz[:, 1], dmp_xyz[:, 2], "--", linewidth=2, label="DMP reproduction")
    ax.plot(exec_xyz[:, 0], exec_xyz[:, 1], exec_xyz[:, 2], linewidth=2, label="Robot executed tool path")

    ax.scatter(demo_xyz[0, 0], demo_xyz[0, 1], demo_xyz[0, 2], s=50, label="Start")
    ax.scatter(demo_xyz[-1, 0], demo_xyz[-1, 1], demo_xyz[-1, 2], s=50, label="Goal")

    ax.set_title("Experiment 1: DMP Replication and Robot Execution")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Error plots
    dmp_point_error, dmp_rms, dmp_final = compute_errors(demo_xyz, dmp_xyz)
    exec_point_error, exec_rms, exec_final = compute_errors(hand_exec, hand_target)

    t1 = np.arange(len(dmp_point_error)) * dt
    t2 = np.arange(len(exec_point_error)) * dt

    plt.figure(figsize=(9, 4))
    plt.plot(t1, dmp_point_error, linewidth=2, label="Demo vs DMP")
    plt.plot(t2, exec_point_error, linewidth=2, label="DMP Hand coordinates vs Executed Hand coordinates")
    plt.title("Tracking Error Through Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print metrics
    print("=== Experiment 1 Metrics ===")
    print(f"DMP final error vs demo      : {dmp_final:.6f}")
    print(f"DMP RMS error vs demo        : {dmp_rms:.6f}")
    print(f"Robot final error vs DMP     : {exec_final:.6f}")
    print(f"Robot RMS error vs DMP       : {exec_rms:.6f}")
    print(f"Demo path length             : {path_length(demo_xyz):.6f}")
    print(f"DMP path length              : {path_length(dmp_xyz):.6f}")
    print(f"Robot executed path length   : {path_length(exec_xyz):.6f}")


# ==================================================
# MAIN EXPERIMENT
# ==================================================
def main():
    t = 5.0 # total duration of the demonstration in seconds
    dt = 0.01 # time step for both DMP reproduction and robot control
    T = int(np.round(t / dt))

    # Tool offset from panda_hand frame
    o_local = np.array([0.00, 0.00, 0.10])

    # Fixed desired hand orientation
    target_orn = p.getQuaternionFromEuler([3.1416, 0.0, 0.0])

    # 1) Create demonstration
    pos = make_demo_path(T=T)   # shape (3, T)

    # 2) Reproduce using DMP
    DMP_pos, DMP_vel, DMP_acc, W, tau_new, newStart, newGoal = dmp_discrete_3d_reproduce(
        pos,
        dt,
        speed=1.0,
        start_shift=np.zeros(3),
        goal_shift=np.zeros(3),
        init_vel_scale=1.0,
        init_vel_offset=np.zeros(3),
    )

    # 3) Send DMP path to robot
    tool_des = DMP_pos.T  # convert from (3, T) to (T, 3)

    tool_exec_log, hand_exec_log, hand_target_log = run_robot_tool_tracking(
        tool_des=tool_des,
        dt=dt,
        o_local=o_local,
        target_orn=target_orn,
        sleep=True
    )

    # 4) Plot performance
    plot_all_results(pos, DMP_pos, tool_exec_log,hand_exec_log, hand_target_log, dt)

    # Keep window open until user closes it manually
    #input("Press Enter to close PyBullet...")
    p.disconnect()


if __name__ == "__main__":
    main()