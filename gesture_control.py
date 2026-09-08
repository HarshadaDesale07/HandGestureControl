import cv2
import mediapipe as mp
import pyautogui

# Get screen size
screen_width, screen_height = pyautogui.size()

# MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Hand detector settings
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

# Start MediaPipe
with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        exit()

    print("Hand Mouse Control Started!")
    print("Move your INDEX FINGER to control the mouse.")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Camera dimensions
        frame_height, frame_width, _ = frame.shape

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Landmark 8 = index fingertip
            index_finger = hand[8]

            # Convert normalized coordinates to camera coordinates
            x = int(index_finger.x * frame_width)
            y = int(index_finger.y * frame_height)

            # Convert camera coordinates to screen coordinates
            screen_x = int(index_finger.x * screen_width)
            screen_y = int(index_finger.y * screen_height)

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Draw fingertip
            cv2.circle(
                frame,
                (x, y),
                10,
                (0, 255, 0),
                -1
            )

            # Display coordinates
            cv2.putText(
                frame,
                f"Mouse: {screen_x}, {screen_y}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Display camera
        cv2.imshow("Gesture Mouse Control", frame)

        # Quit with Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Gesture control stopped.")