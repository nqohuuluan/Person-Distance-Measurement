import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize MediaPipe Pose Detection API
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load video file
cap = cv2.VideoCapture('2walking_rz.mp4')

# Select the person to track
person_id = 1

while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect poses in the frame
    results = pose.process(frame)

    # Extract landmarks for the person to track
    landmarks = []
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i % results.pose_landmarks.landmark_count == person_id:
                landmarks.append((landmark.x, landmark.y, landmark.z))

    # Track the person using the extracted landmarks
    # TODO: Implement tracking algorithm




    # Draw pose landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the frame
    cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()