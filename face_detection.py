# Face Detection using OpenCV
import cv2

# Load the face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open the webcam
camera = cv2.VideoCapture(0)

while True:
    # Capture video frame
    ret, frame = camera.read()

    if not ret:
        print("Could not access the camera.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Draw rectangle around every face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # Display number of faces
    cv2.putText(
        frame,
        "Faces: " + str(len(faces)),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show the camera
    cv2.imshow("Face Detector", frame)

    # Press Q to close
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close camera and window
camera.release()
cv2.destroyAllWindows()