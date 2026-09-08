import cv2
import mediapipe as mp

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Hand connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# MediaPipe configuration
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

# Start hand detector
with HandLandmarker.create_from_options(options) as landmarker:

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        exit()

    print("Hand Gesture Detection Started!")
    print("Press Q to quit.")

    while True:

        # Read webcam frame
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam.")
            break

        # Mirror the camera
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detect hands
        result = landmarker.detect(mp_image)

        # Draw detected hands
        if result.hand_landmarks:

            for hand in result.hand_landmarks:

                points = []

                # Draw landmark points
                for landmark in hand:

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    points.append((x, y))

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw hand skeleton
                for start, end in CONNECTIONS:

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 0, 0),
                        2
                    )

        # Display camera
        cv2.imshow("Hand Gesture Control", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release webcam
    cap.release()
    cv2.destroyAllWindows()

print("Program ended.")