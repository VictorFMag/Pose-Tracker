import argparse
import json
import socket
import time

import pybullet as p


UDP_HOST = "127.0.0.1"
UDP_PORT = 5005
UPDATE_FPS = 120
POSITION_SMOOTHING = 0.35
BODY_HEIGHT_OFFSET = 1.05
BODY_SCALE = 1.4
MIN_VISIBILITY = 0.45

# MediaPipe Pose landmark indexes.
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT = 31
RIGHT_FOOT = 32

SKELETON_CONNECTIONS = [
    (LEFT_EAR, NOSE),
    (NOSE, RIGHT_EAR),
    (LEFT_EAR, LEFT_SHOULDER),
    (RIGHT_EAR, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (LEFT_WRIST, LEFT_PINKY),
    (LEFT_WRIST, LEFT_INDEX),
    (LEFT_WRIST, LEFT_THUMB),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (RIGHT_WRIST, RIGHT_PINKY),
    (RIGHT_WRIST, RIGHT_INDEX),
    (RIGHT_WRIST, RIGHT_THUMB),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT),
    (LEFT_ANKLE, LEFT_FOOT),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT),
    (RIGHT_ANKLE, RIGHT_FOOT),
]

MAIN_JOINTS = {
    NOSE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT,
    RIGHT_FOOT,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render MediaPipe world landmarks as a PyBullet mannequin."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=UDP_PORT,
        help=f"UDP port used to receive landmarks. Default: {UDP_PORT}.",
    )
    return parser.parse_args()


def create_udp_socket(port: int) -> socket.socket:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind((UDP_HOST, port))
    receiver.setblocking(False)
    return receiver


def receive_latest_packet(receiver: socket.socket) -> dict | None:
    latest_packet = None

    while True:
        try:
            raw_packet, _ = receiver.recvfrom(65535)
        except BlockingIOError:
            break

        try:
            packet = json.loads(raw_packet.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

        if isinstance(packet, dict):
            latest_packet = packet

    return latest_packet


def convert_landmark(landmark: list[float]) -> list[float]:
    x, y, z = landmark[:3]

    # MediaPipe: X right, Y down, Z depth.
    # PyBullet: X right, Y depth, Z up.
    return [
        BODY_SCALE * x,
        -BODY_SCALE * z,
        BODY_HEIGHT_OFFSET - BODY_SCALE * y,
    ]


def convert_landmarks(raw_landmarks: list) -> tuple[list[list[float]], list[float]]:
    positions = []
    visibility = []

    for landmark in raw_landmarks:
        if not isinstance(landmark, list) or len(landmark) < 4:
            raise ValueError("Invalid landmark packet.")

        positions.append(convert_landmark(landmark))
        visibility.append(float(landmark[3]))

    return positions, visibility


def smooth_positions(
    current: list[list[float]] | None,
    target: list[list[float]],
) -> list[list[float]]:
    if current is None or len(current) != len(target):
        return [position.copy() for position in target]

    for current_position, target_position in zip(current, target):
        for axis in range(3):
            current_position[axis] += (
                target_position[axis] - current_position[axis]
            ) * POSITION_SMOOTHING

    return current


def create_joint_markers() -> dict[int, int]:
    marker_shape = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.035,
        rgbaColor=[1.0, 0.75, 0.1, 1.0],
    )

    markers = {}

    for landmark_index in MAIN_JOINTS:
        markers[landmark_index] = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=marker_shape,
            basePosition=[0.0, 0.0, -10.0],
        )

    return markers


def update_joint_markers(
    markers: dict[int, int],
    positions: list[list[float]],
    visibility: list[float],
) -> None:
    for landmark_index, body_id in markers.items():
        if visibility[landmark_index] >= MIN_VISIBILITY:
            position = positions[landmark_index]
        else:
            position = [0.0, 0.0, -10.0]

        p.resetBasePositionAndOrientation(
            body_id,
            position,
            [0.0, 0.0, 0.0, 1.0],
        )


def update_bones(
    line_ids: list[int],
    positions: list[list[float]],
    visibility: list[float],
) -> None:
    for connection_index, (start_index, end_index) in enumerate(
        SKELETON_CONNECTIONS
    ):
        previous_line_id = line_ids[connection_index]

        if min(visibility[start_index], visibility[end_index]) < MIN_VISIBILITY:
            if previous_line_id >= 0:
                p.removeUserDebugItem(previous_line_id)
                line_ids[connection_index] = -1
            continue

        line_ids[connection_index] = p.addUserDebugLine(
            lineFromXYZ=positions[start_index],
            lineToXYZ=positions[end_index],
            lineColorRGB=[0.1, 0.85, 1.0],
            lineWidth=7.0,
            lifeTime=0,
            replaceItemUniqueId=previous_line_id,
        )


def add_scene_references() -> None:
    p.addUserDebugLine(
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.4, 0.4, 0.4],
        1.0,
    )
    p.addUserDebugLine(
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.4, 0.4, 0.4],
        1.0,
    )


def main() -> None:
    arguments = parse_arguments()
    client_id = p.connect(p.GUI)

    if client_id < 0:
        raise RuntimeError("Could not start the PyBullet GUI.")

    receiver = create_udp_socket(arguments.port)

    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=2.6,
            cameraYaw=0.0,
            cameraPitch=-5.0,
            cameraTargetPosition=[0.0, 0.0, 1.0],
        )
        p.setGravity(0.0, 0.0, 0.0)

        add_scene_references()
        markers = create_joint_markers()
        line_ids = [-1] * len(SKELETON_CONNECTIONS)
        current_positions = None
        current_visibility = [0.0] * 33
        connected = False
        time_step = 1.0 / UPDATE_FPS

        print(f"Waiting for world landmarks on udp://{UDP_HOST}:{arguments.port}")
        print("Start pose_tracker.py in another terminal.")

        while p.isConnected():
            frame_start = time.perf_counter()
            packet = receive_latest_packet(receiver)

            if packet is not None:
                raw_landmarks = packet.get("world_landmarks")

                if isinstance(raw_landmarks, list) and len(raw_landmarks) == 33:
                    try:
                        target_positions, current_visibility = convert_landmarks(
                            raw_landmarks
                        )
                    except (TypeError, ValueError):
                        target_positions = None

                    if target_positions is not None:
                        current_positions = smooth_positions(
                            current_positions,
                            target_positions,
                        )

                        if not connected:
                            print(
                                "Full skeleton received. "
                                "Atom Budget Edition is online."
                            )
                            connected = True

            if current_positions is not None:
                update_joint_markers(
                    markers,
                    current_positions,
                    current_visibility,
                )
                update_bones(
                    line_ids,
                    current_positions,
                    current_visibility,
                )

            p.stepSimulation()

            elapsed = time.perf_counter() - frame_start
            remaining = time_step - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
