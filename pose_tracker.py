import json
import math
import socket
import time

import cv2
import mediapipe as mp


UDP_HOST = "127.0.0.1"
UDP_PORT = 5005


def calculate_angle(first, middle, last) -> float:
    vector_a_x = first.x - middle.x
    vector_a_y = first.y - middle.y
    vector_b_x = last.x - middle.x
    vector_b_y = last.y - middle.y

    dot_product = vector_a_x * vector_b_x + vector_a_y * vector_b_y
    magnitude_a = math.hypot(vector_a_x, vector_a_y)
    magnitude_b = math.hypot(vector_b_x, vector_b_y)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    cosine = dot_product / (magnitude_a * magnitude_b)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def get_joint_angle(landmarks, joint_indexes) -> float | None:
    first = landmarks[joint_indexes[0]]
    middle = landmarks[joint_indexes[1]]
    last = landmarks[joint_indexes[2]]

    if min(first.visibility, middle.visibility, last.visibility) < 0.5:
        return None

    return calculate_angle(first, middle, last)


def get_head_tilt(landmarks, mp_pose) -> float | None:
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    required_landmarks = [
        left_ear,
        right_ear,
        left_shoulder,
        right_shoulder,
        nose,
    ]

    if min(point.visibility for point in required_landmarks) < 0.5:
        return None

    ear_angle = math.degrees(
        math.atan2(
            right_ear.y - left_ear.y,
            right_ear.x - left_ear.x,
        )
    )
    shoulder_angle = math.degrees(
        math.atan2(
            right_shoulder.y - left_shoulder.y,
            right_shoulder.x - left_shoulder.x,
        )
    )

    head_tilt = ear_angle - shoulder_angle
    return (head_tilt + 180.0) % 360.0 - 180.0


def draw_value(
    frame,
    landmark,
    label: str,
    value: float,
    color: tuple[int, int, int],
) -> None:
    frame_height, frame_width = frame.shape[:2]
    position = (
        int(landmark.x * frame_width) + 10,
        int(landmark.y * frame_height) - 10,
    )

    cv2.putText(
        frame,
        f"{label}: {value:.0f} deg",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def send_pose_data(sender: socket.socket, pose_data: dict[str, object]) -> None:
    packet = json.dumps(pose_data).encode("utf-8")
    sender.sendto(packet, (UDP_HOST, UDP_PORT))


def serialize_world_landmarks(world_landmarks) -> list[list[float]]:
    return [
        [
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility,
        ]
        for landmark in world_landmarks
    ]


def main() -> None:
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Keep this order so DirectShow selects MJPG at 60 FPS.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 60)
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    print("Width:", camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Camera FPS:", camera.get(cv2.CAP_PROP_FPS))
    print(f"Sending pose data to udp://{UDP_HOST}:{UDP_PORT}")

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    joint_definitions = {
        "left_elbow": (
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
        ),
        "right_elbow": (
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST,
        ),
        "left_knee": (
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
        ),
        "right_knee": (
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
        ),
    }

    display_labels = {
        "left_elbow": "L elbow",
        "right_elbow": "R elbow",
        "left_knee": "L knee",
        "right_knee": "R knee",
    }

    previous_time = time.perf_counter()

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                result = pose.process(rgb_frame)
                rgb_frame.flags.writeable = True

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    pose_data = {}

                    mp_drawing.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 255),
                            thickness=2,
                            circle_radius=3,
                        ),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(255, 80, 80),
                            thickness=2,
                        ),
                    )

                    for name, indexes in joint_definitions.items():
                        angle = get_joint_angle(landmarks, indexes)

                        if angle is None:
                            continue

                        pose_data[name] = angle
                        draw_value(
                            frame,
                            landmarks[indexes[1]],
                            display_labels[name],
                            angle,
                            (0, 255, 0),
                        )

                    head_tilt = get_head_tilt(landmarks, mp_pose)
                    if head_tilt is not None:
                        pose_data["head_tilt"] = head_tilt
                        draw_value(
                            frame,
                            landmarks[mp_pose.PoseLandmark.NOSE],
                            "Head",
                            head_tilt,
                            (255, 255, 0),
                        )

                    if result.pose_world_landmarks:
                        pose_data["world_landmarks"] = serialize_world_landmarks(
                            result.pose_world_landmarks.landmark
                        )

                    if pose_data:
                        send_pose_data(sender, pose_data)

                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                previous_time = current_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                cv2.putText(
                    frame,
                    f"FPS: {fps:.0f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "UDP: 127.0.0.1:5005",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Atom Budget Edition - Pose Tracker", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        sender.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()