import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        h, w, _ = frame.shape

        # Thumb tip
        thumb = hand.landmark[4]
        thumb_x = int(thumb.x * w)
        thumb_y = int(thumb.y * h)

        # Index finger tip
        index = hand.landmark[8]
        index_x = int(index.x * w)
        index_y = int(index.y * h)

        # Distance between thumb and index
        distance = math.hypot(
            index_x - thumb_x,
            index_y - thumb_y
        )

        # Draw line between fingers
        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (index_x, index_y),
            (255, 0, 255),
            3
        )

        # Volume control
        if distance < 40:
            pyautogui.press("volumedown")

        elif distance > 150:
            pyautogui.press("volumeup")

        # Display distance
        cv2.putText(
            frame,
            f"Pinch Distance: {int(distance)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow("Volume Control", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()