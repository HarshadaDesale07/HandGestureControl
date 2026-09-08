import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Screen size
screen_width, screen_height = pyautogui.size()

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

# Prevent repeated clicks
last_click_time = 0
click_cooldown = 0.7

with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        exit()

    print("Pinch Control Started!")
    print("🤏 Pinch thumb + index finger = LEFT CLICK")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Index fingertip = landmark 8
            index = hand[8]

            # Thumb tip = landmark 4
            thumb = hand[4]

            # Camera coordinates
            index_x = int(index.x * frame_width)
            index_y = int(index.y * frame_height)

            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)

            # Distance between thumb and index finger
            distance = math.sqrt(
                (index_x - thumb_x) ** 2 +
                (index_y - thumb_y) ** 2
            )

            # Draw both fingertips
            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (255, 0, 0),
                -1
            )

            # Draw line between thumb and index
            cv2.line(
                frame,
                (index_x, index_y),
                (thumb_x, thumb_y),
                (255, 255, 255),
                2
            )

            # Pinch detection
            if distance < 40:

                cv2.putText(
                    frame,
                    "PINCH - CLICK",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                # Click only once per pinch
                current_time = time.time()

                if current_time - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = current_time

            else:

                cv2.putText(
                    frame,
                    "OPEN",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

            # Show distance
            cv2.putText(
                frame,
                f"Distance: {int(distance)}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Display webcam
        cv2.imshow("Pinch Control", frame)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Pinch control stopped.")