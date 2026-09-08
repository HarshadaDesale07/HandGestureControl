import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ---------------- SETTINGS ----------------
MODEL_PATH = "models/hand_landmarker.task"

SCREEN_W, SCREEN_H = pyautogui.size()

smooth_factor = 0.25

PINCH_THRESHOLD = 40
PINCH_COOLDOWN = 0.7

# Continuous scrolling
SCROLL_AMOUNT = 6
SCROLL_DELAY = 0.05


# ---------------- COLORS ----------------
GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)
BLUE = (255, 150, 0)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)


# ---------------- MEDIAPIPE ----------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)


# ---------------- FUNCTIONS ----------------

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    ) * 1000


def get_finger_states(hand):
    index = hand[8].y < hand[6].y
    middle = hand[12].y < hand[10].y
    ring = hand[16].y < hand[14].y
    pinky = hand[20].y < hand[18].y

    return index, middle, ring, pinky


# ---------------- START CAMERA ----------------

cap = cv2.VideoCapture(0)

previous_x = SCREEN_W // 2
previous_y = SCREEN_H // 2

last_click_time = 0
last_scroll_time = 0


# ---------------- MAIN PROGRAM ----------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        gesture_text = "NO HAND"
        gesture_color = WHITE


        # =================================================
        # HAND DETECTED
        # =================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # ---------------- FINGER STATES ----------------

            index, middle, ring, pinky = get_finger_states(hand)


            # ---------------- PINCH ----------------

            pinch_distance = distance(hand[4], hand[8])

            pinch = pinch_distance < PINCH_THRESHOLD


            # ---------------- GESTURES ----------------

            two_fingers = (
                index
                and middle
                and not ring
                and not pinky
            )

            fist = (
                not index
                and not middle
                and not ring
                and not pinky
                and not pinch
            )

            open_palm = (
                index
                and middle
                and ring
                and pinky
            )


            # =================================================
            # PINCH → LEFT CLICK
            # =================================================

            if pinch:

                gesture_text = "PINCH  |  CLICK"
                gesture_color = PURPLE

                current_time = time.time()

                if current_time - last_click_time > PINCH_COOLDOWN:

                    pyautogui.click()

                    last_click_time = current_time


            # =================================================
            # TWO FINGERS → CONTINUOUS SCROLL UP
            # =================================================

            elif two_fingers:

                gesture_text = "TWO FINGERS  |  SCROLL UP"
                gesture_color = BLUE

                current_time = time.time()

                if current_time - last_scroll_time > SCROLL_DELAY:

                    pyautogui.scroll(SCROLL_AMOUNT)

                    last_scroll_time = current_time


            # =================================================
            # FIST → CONTINUOUS SCROLL DOWN
            # =================================================

            elif fist:

                gesture_text = "FIST  |  SCROLL DOWN"
                gesture_color = ORANGE

                current_time = time.time()

                if current_time - last_scroll_time > SCROLL_DELAY:

                    pyautogui.scroll(-SCROLL_AMOUNT)

                    last_scroll_time = current_time


            # =================================================
            # OPEN PALM → STOP
            # =================================================

            elif open_palm:

                gesture_text = "OPEN PALM  |  STOP"
                gesture_color = WHITE


            # =================================================
            # INDEX ONLY → MOUSE CONTROL
            # =================================================

            elif index and not middle and not ring and not pinky:

                gesture_text = "INDEX  |  MOUSE"
                gesture_color = GREEN

                x = int(hand[8].x * SCREEN_W)
                y = int(hand[8].y * SCREEN_H)

                # Smooth movement

                current_x = previous_x + int(
                    (x - previous_x) * smooth_factor
                )

                current_y = previous_y + int(
                    (y - previous_y) * smooth_factor
                )

                pyautogui.moveTo(
                    current_x,
                    current_y
                )

                previous_x = current_x
                previous_y = current_y


            else:

                gesture_text = "UNKNOWN"
                gesture_color = RED


            # =================================================
            # DRAW HAND LANDMARKS
            # =================================================

            for landmark in hand:

                x = int(
                    landmark.x *
                    frame.shape[1]
                )

                y = int(
                    landmark.y *
                    frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    gesture_color,
                    -1
                )


        # =================================================
        # GESTURE STATUS BOX
        # =================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (500, 65),
            (20, 20, 20),
            -1
        )

        cv2.rectangle(
            frame,
            (10, 10),
            (500, 65),
            gesture_color,
            2
        )

        cv2.putText(
            frame,
            gesture_text,
            (25, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            gesture_color,
            2
        )


        # =================================================
        # CONTROLS
        # =================================================

        cv2.putText(
            frame,
            "Q = QUIT",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2
        )


        # =================================================
        # SHOW CAMERA
        # =================================================

        cv2.imshow(
            "Hand Gesture Control",
            frame
        )


        # =================================================
        # QUIT
        # =================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ---------------- CLEANUP ----------------

cap.release()
cv2.destroyAllWindows()