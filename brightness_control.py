import cv2
import mediapipe as mp
import screen_brightness_control as sbc

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

        # Index finger tip
        index = hand.landmark[8]

        index_x = int(index.x * w)
        index_y = int(index.y * h)

        # Convert index finger Y position to brightness
        brightness = int(
            max(0, min(100, (400 - index_y) * 100 / 300))
        )

        # Set brightness
        try:
            sbc.set_brightness(brightness)
        except:
            pass

        # Show index finger position
        cv2.circle(
            frame,
            (index_x, index_y),
            10,
            (255, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"Brightness: {brightness}%",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow("Brightness Control", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
