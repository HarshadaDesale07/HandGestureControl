import cv2
import mediapipe as mp
import pyautogui
import time

# ==========================================
# MediaPipe Setup
# ==========================================

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

# ==========================================
# Settings
# ==========================================

previous_y = None

# Ignore tiny movements
movement_threshold = 15

# Scroll strength
scroll_speed = 1.5

# Prevent excessive scrolling
last_scroll_time = 0
scroll_delay = 0.05


# ==========================================
# Finger Detection
# ==========================================

def get_finger_states(hand):
    """
    Returns the state of:
    Index, Middle, Ring, Pinky

    True  = finger extended
    False = finger folded
    """

    index = hand[8].y < hand[6].y
    middle = hand[12].y < hand[10].y
    ring = hand[16].y < hand[14].y
    pinky = hand[20].y < hand[18].y

    return index, middle, ring, pinky


# ==========================================
# Start MediaPipe
# ==========================================

with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        exit()

    print()
    print("======================================")
    print("       GESTURE SCROLL CONTROL")
    print("======================================")
    print("✌️  Two fingers + MOVE UP   = SCROLL UP")
    print("✊  Fist + MOVE DOWN        = SCROLL DOWN")
    print("✋  Open palm               = NO SCROLL")
    print("Press Q to quit")
    print("======================================")
    print()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape

        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Get finger states
            index, middle, ring, pinky = get_finger_states(hand)

            # ======================================
            # GESTURE 1: TWO FINGERS ✌️
            # ======================================

            two_fingers = (
                index
                and middle
                and not ring
                and not pinky
            )

            # ======================================
            # GESTURE 2: FIST ✊
            # ======================================

            fist = (
                not index
                and not middle
                and not ring
                and not pinky
            )

            # ======================================
            # GESTURE 3: OPEN PALM ✋
            # ======================================

            open_palm = (
                index
                and middle
                and ring
                and pinky
            )

            # ======================================
            # TWO FINGERS → SCROLL UP
            # ======================================

            if two_fingers:

                # Use midpoint between index and middle
                center_y = int(
                    (
                        (hand[8].y + hand[12].y) / 2
                    ) * frame_height
                )

                center_x = int(
                    (
                        (hand[8].x + hand[12].x) / 2
                    ) * frame_width
                )

                # Draw tracking point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    10,
                    (0, 255, 0),
                    -1
                )

                if previous_y is None:

                    previous_y = center_y

                else:

                    movement = previous_y - center_y

                    current_time = time.time()

                    # --------------------------------
                    # MOVE UP → SCROLL UP
                    # --------------------------------

                    if (
                        movement > movement_threshold
                        and current_time - last_scroll_time > scroll_delay
                    ):

                        scroll_amount = max(
                            1,
                            int(movement * scroll_speed)
                        )

                        pyautogui.scroll(scroll_amount)

                        cv2.putText(
                            frame,
                            "SCROLL UP",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )

                        last_scroll_time = current_time

                    previous_y = center_y

                cv2.putText(
                    frame,
                    "✌️  SCROLL UP MODE",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # ======================================
            # FIST → SCROLL DOWN
            # ======================================

            elif fist:

                # Use wrist as tracking point
                center_y = int(
                    hand[0].y * frame_height
                )

                center_x = int(
                    hand[0].x * frame_width
                )

                # Draw tracking point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    10,
                    (0, 0, 255),
                    -1
                )

                if previous_y is None:

                    previous_y = center_y

                else:

                    movement = previous_y - center_y

                    current_time = time.time()

                    # --------------------------------
                    # MOVE DOWN → SCROLL DOWN
                    # --------------------------------

                    if (
                        movement < -movement_threshold
                        and current_time - last_scroll_time > scroll_delay
                    ):

                        scroll_amount = max(
                            1,
                            int(abs(movement) * scroll_speed)
                        )

                        pyautogui.scroll(-scroll_amount)

                        cv2.putText(
                            frame,
                            "SCROLL DOWN",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2
                        )

                        last_scroll_time = current_time

                    previous_y = center_y

                cv2.putText(
                    frame,
                    "✊  SCROLL DOWN MODE",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # ======================================
            # OPEN PALM → NOTHING
            # ======================================

            elif open_palm:

                previous_y = None

                cv2.putText(
                    frame,
                    "✋  NO SCROLL",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

            # ======================================
            # OTHER GESTURES → NOTHING
            # ======================================

            else:

                previous_y = None

                cv2.putText(
                    frame,
                    "NO SCROLL",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

        else:

            # No hand detected
            previous_y = None

        # ==========================================
        # Display Camera
        # ==========================================

        cv2.imshow(
            "Gesture Scroll Control",
            frame
        )

        # Q → Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ==========================================
    # Cleanup
    # ==========================================

    cap.release()
    cv2.destroyAllWindows()

print("Scroll control stopped.")