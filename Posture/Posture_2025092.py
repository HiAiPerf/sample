import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_posture_and_gesture(landmarks, image_shape):
    h, w, _ = image_shape
    feedback = []

    # Get key points (shoulders and nose for posture)
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # Convert normalized coords to pixels
    ls = np.array([left_shoulder.x * w, left_shoulder.y * h])
    rs = np.array([right_shoulder.x * w, right_shoulder.y * h])
    nose_pt = np.array([nose.x * w, nose.y * h])

    # --- Posture analysis (nose aligned with shoulders) ---
    shoulder_mid = (ls + rs) / 2
    vertical_diff = abs(nose_pt[0] - shoulder_mid[0])

    if vertical_diff < 40:
        feedback.append("✅ Upright posture")
    else:
        feedback.append("⚠️ Check posture (head tilted/leaning)")

    # --- Gesture openness (shoulder distance vs torso width) ---
    shoulder_distance = np.linalg.norm(ls - rs)
    torso_height = abs(nose_pt[1] - ((ls[1] + rs[1]) / 2))

    if shoulder_distance > torso_height * 0.7:
        feedback.append("✅ Open gestures")
    else:
        feedback.append("⚠️ Gestures too closed")

    return feedback

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for natural viewing
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Analyze
        feedback = analyze_posture_and_gesture(results.pose_landmarks.landmark, frame.shape)

        # Show feedback on screen
        y0 = 30
        for i, text in enumerate(feedback):
            cv2.putText(frame, text, (10, y0 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Posture Analysis", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
