import sys
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import cvzone

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# Mediapipe for face detection
import mediapipe as mp
import time

# mp_facedetector = mp.solutions.face_detection
# mp_draw = mp.solutions.drawing_utils

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


# Open both cameras
cap_right = cv2.VideoCapture(2)
cap_left =  cv2.VideoCapture(0)

# cap_right = cv2.VideoCapture("2walking_rz.mp4")
# cap_left =  cv2.VideoCapture("2walking_rz.mp4")


# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 9               #Distance between the cameras [cm]
f = 8              #Camera lense's focal length [mm]
alpha = 56.6        #Camera field of view in the horisontal plane [degrees]




# Main program loop with face detector and depth estimation using stereo vision
# with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

################## CALIBRATION #########################################################

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

########################################################################################

    # If cannot catch any frame, break
    if not succes_right or not succes_left:
        break

    else:

        start = time.time()

        # Convert the BGR image to RGB
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        results_right = pose.process(frame_right)
        results_left = pose.process(frame_left)

        # Convert the RGB image to BGR
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)


        ################## CALCULATING DEPTH #########################################################

        center_right = 0
        center_left = 0
        point1 = 0
        point2 = 0
        cx_point1 = 0
        cx_point2 = 0
        cy_point1 = 0
        cy_point2 = 0
        distance = 0


        if results_right.pose_landmarks:
            mpDraw.draw_landmarks(frame_right, results_right.pose_landmarks, mpPose.POSE_CONNECTIONS)
            h, w, c = frame_right.shape  # height, width, chanel của ảnh
            for id, lm in enumerate(results_right.pose_landmarks.landmark):
                if id == 11:  # id = 11 và id = 12 là id của hai điểm trên vai
                    point1 = lm.x
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_right, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    cx_point1 = cx
                    cy_point1 = cy
                    # print(id, cx)
                    # print(id, lm.x)
                if id == 12:
                    point2 = lm.x
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_right, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    cx_point2 = cx
                    cy_point2 = cy
                distance = point1 - point2

                if distance != 0:
                    distance_pixel = int(distance * w)  # distance*w: [khoảng cách theo tỉ lệ 1:1 (pixel)] * [chiều dài khung hình (640)]
                    midpointR_x = int(cx_point1 - (cx_point1 - cx_point2) / 2)
                    midpointR_y = int(cy_point1 - (cy_point1 - cy_point2) / 2)
                    cv2.circle(frame_right, (midpointR_x, midpointR_y), 5, (255, 0, 0), cv2.FILLED)
                    # print("midpoint: ", midpointR_x)

        if results_left.pose_landmarks:
            mpDraw.draw_landmarks(frame_left, results_left.pose_landmarks, mpPose.POSE_CONNECTIONS)
            h, w, c = frame_left.shape  # height, width, chanel của ảnh
            for id, lm in enumerate(results_left.pose_landmarks.landmark):
                if id == 11:  # id = 11 và id = 12 là id của hai điểm trên vai
                    point1 = lm.x
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_left, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    cx_point1 = cx
                    cy_point1 = cy
                    # print(id, cx)
                    # print(id, lm.x)
                if id == 12:
                    point2 = lm.x
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_left, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    cx_point2 = cx
                    cy_point2 = cy
                distance = point1 - point2

                if distance != 0:
                    distance_pixel = int(distance * w)  # distance*w: [khoảng cách theo tỉ lệ 1:1 (pixel)] * [chiều dài khung hình (640)]
                    midpointL_x = int(cx_point1 - (cx_point1 - cx_point2) / 2)
                    midpointL_y = int(cy_point1 - (cy_point1 - cy_point2) / 2)
                    cv2.circle(frame_left, (midpointL_x, midpointL_y), 5, (255, 0, 0), cv2.FILLED)
                    # print("midpoint: ", midpointL_x)

        # If no ball can be caught in one camera show text "TRACKING LOST"
        if not results_right.pose_landmarks or not results_left.pose_landmarks:
            cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        else:
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
            # All formulas used to find depth is in video presentaion
            depth = tri.find_depth((midpointR_x,midpointR_y), (midpointL_x, midpointL_y), frame_right, frame_left, B, f, alpha)

            cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            #print("Depth: ", str(round(depth,1)))
            turnLR = 320 - midpointR_x
            if turnLR > 0:
                print("turn left", turnLR)
            else: # turnLR < 0:
                print("turn right: ", turnLR)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        #print("FPS: ", fps)

        cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)


        # Show the frames
        cv2.imshow("frame right", frame_right)
        cv2.imshow("frame left", frame_left)


        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()