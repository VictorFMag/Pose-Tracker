import argparse
import json
import math
import socket
import time

import pybullet as p
import pybullet_data


UDP_HOST = "127.0.0.1"
UDP_PORT = 5005
SIMULATION_FPS = 240
SMOOTHING = 0.25

DEFAULT_ANGLES = {
    "left_elbow": 180.0,
    "right_elbow": 180.0,
    "left_knee": 180.0,
    "right_knee": 180.0,
    "head_tilt": 0.0,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control a PyBullet humanoid using body angles received over UDP."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Animate the humanoid without waiting for pose tracker data.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=UDP_PORT,
        help=f"UDP port used to receive angles. Default: {UDP_PORT}.",
    )
    return parser.parse_args()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def smooth_angles(current: dict[str, float], target: dict[str, float]) -> None:
    for name in current:
        current[name] += (target[name] - current[name]) * SMOOTHING


def create_udp_socket(port: int) -> socket.socket:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind((UDP_HOST, port))
    receiver.setblocking(False)
    return receiver


def receive_latest_angles(
    receiver: socket.socket,
    target_angles: dict[str, float],
) -> bool:
    received_packet = False

    while True:
        try:
            packet, _ = receiver.recvfrom(4096)
        except BlockingIOError:
            break

        try:
            data = json.loads(packet.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

        for name in target_angles:
            value = data.get(name)
            if isinstance(value, (int, float)):
                target_angles[name] = float(value)

        received_packet = True

    return received_packet


def create_joint_map(humanoid_id: int) -> dict[str, int]:
    joints = {}

    for joint_index in range(p.getNumJoints(humanoid_id)):
        joint_info = p.getJointInfo(humanoid_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")
        joints[joint_name] = joint_index

    required_joints = {
        "neck",
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
    }

    missing_joints = required_joints - joints.keys()
    if missing_joints:
        missing = ", ".join(sorted(missing_joints))
        raise RuntimeError(f"Humanoid model is missing joints: {missing}")

    return joints


def apply_pose(
    humanoid_id: int,
    joints: dict[str, int],
    angles: dict[str, float],
) -> None:
    left_elbow_flexion = math.radians(
        180.0 - clamp(angles["left_elbow"], 0.0, 180.0)
    )
    right_elbow_flexion = math.radians(
        180.0 - clamp(angles["right_elbow"], 0.0, 180.0)
    )

    left_knee_flexion = -math.radians(
        180.0 - clamp(angles["left_knee"], 0.0, 180.0)
    )
    right_knee_flexion = -math.radians(
        180.0 - clamp(angles["right_knee"], 0.0, 180.0)
    )

    head_tilt = math.radians(clamp(angles["head_tilt"], -60.0, 60.0))
    head_orientation = p.getQuaternionFromEuler([head_tilt, 0.0, 0.0])

    p.resetJointState(
        humanoid_id,
        joints["left_elbow"],
        left_elbow_flexion,
    )
    p.resetJointState(
        humanoid_id,
        joints["right_elbow"],
        right_elbow_flexion,
    )
    p.resetJointState(
        humanoid_id,
        joints["left_knee"],
        left_knee_flexion,
    )
    p.resetJointState(
        humanoid_id,
        joints["right_knee"],
        right_knee_flexion,
    )
    p.resetJointStateMultiDof(
        humanoid_id,
        joints["neck"],
        targetValue=head_orientation,
        targetVelocity=[0.0, 0.0, 0.0],
    )


def update_demo_angles(target_angles: dict[str, float], elapsed: float) -> None:
    target_angles["left_elbow"] = 105.0 + 70.0 * math.sin(elapsed * 1.4)
    target_angles["right_elbow"] = 105.0 + 70.0 * math.sin(
        elapsed * 1.4 + math.pi
    )
    target_angles["left_knee"] = 145.0 + 30.0 * math.sin(elapsed * 1.1)
    target_angles["right_knee"] = 145.0 + 30.0 * math.sin(
        elapsed * 1.1 + math.pi
    )
    target_angles["head_tilt"] = 25.0 * math.sin(elapsed * 0.9)


def main() -> None:
    arguments = parse_arguments()

    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Could not start the PyBullet GUI.")

    receiver = None

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=2.8,
            cameraYaw=0.0,
            cameraPitch=0.0,
            cameraTargetPosition=[0.0, 1.0, 0.0],
        )

        humanoid_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            basePosition=[0.0, 0.9, 0.0],
            globalScaling=0.25,
            useFixedBase=True,
            flags=p.URDF_MAINTAIN_LINK_ORDER,
        )

        joints = create_joint_map(humanoid_id)
        current_angles = DEFAULT_ANGLES.copy()
        target_angles = DEFAULT_ANGLES.copy()

        if arguments.demo:
            print("Demo mode enabled. Close the PyBullet window to stop.")
        else:
            receiver = create_udp_socket(arguments.port)
            print(f"Waiting for pose data on udp://{UDP_HOST}:{arguments.port}")
            print("Close the PyBullet window or press Ctrl+C to stop.")

        start_time = time.perf_counter()
        last_packet_time = None
        time_step = 1.0 / SIMULATION_FPS

        while p.isConnected():
            frame_start = time.perf_counter()

            if arguments.demo:
                update_demo_angles(
                    target_angles,
                    frame_start - start_time,
                )
            elif receiver is not None and receive_latest_angles(
                receiver,
                target_angles,
            ):
                if last_packet_time is None:
                    print("Pose data received. Atom Budget Edition is online.")
                last_packet_time = frame_start

            smooth_angles(current_angles, target_angles)
            apply_pose(humanoid_id, joints, current_angles)
            p.stepSimulation()

            elapsed = time.perf_counter() - frame_start
            remaining = time_step - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass
    finally:
        if receiver is not None:
            receiver.close()
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
