import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
# Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


def main():
    cap_right = cv2.VideoCapture('test_video.mp4')
    cap_left = cv2.VideoCapture('test_video.mp4')

    # cap_right = cv2.VideoCapture(0)
    # cap_left = cv2.VideoCapture(2)

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True:
        ret_right, frame_right = cap_right.read()
        ret_left, frame_left = cap_left.read()

        frame_right = imutils.resize(frame_right, width=600)
        frame_left = imutils.resize(frame_left, width=600)

        total_frames = total_frames + 1

        (H, W) = frame_right.shape[:2]

        blob_right = cv2.dnn.blobFromImage(frame_right, 0.007843, (W, H), 127.5)
        blob_left = cv2.dnn.blobFromImage(frame_left, 0.007843, (W, H), 127.5)

        detector.setInput(blob_right)
        detector.setInput(blob_left)

        person_detections_right = detector.forward()
        person_detections_left = detector.forward()

        rects_right = []
        rects_left = []
        if ret_right:
            for i in np.arange(0, person_detections_right.shape[2]):
                confidence = person_detections_right[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(person_detections_right[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    person_box_right = person_detections_right[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = person_box_right.astype("int")
                    rects_right.append(person_box_right)

        if ret_left:
            for i in np.arange(0, person_detections_left.shape[2]):
                confidence = person_detections_left[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(person_detections_left[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    person_box_left = person_detections_left[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = person_box_left.astype("int")
                    rects_left.append(person_box_left)


        boundingboxes_right = np.array(rects_right)
        boundingboxes_right = boundingboxes_right.astype(int)
        rects_right = non_max_suppression_fast(boundingboxes_right, 0.3)

        boundingboxes_left = np.array(rects_left)
        boundingboxes_left = boundingboxes_left.astype(int)
        rects_left = non_max_suppression_fast(boundingboxes_left, 0.3)

        objects_right = tracker.update(rects_right)
        objects_left = tracker.update(rects_left)

        for (objectId_R, bbox_R), (objectId_L, bbox_L) in zip(objects_right.items(), objects_left.items()):
            x1R, y1R, x2R, y2R = bbox_R
            x1L, y1L, x2L, y2L = bbox_L

            x1R = int(x1R)
            y1R = int(y1R)
            x2R = int(x2R)
            y2R = int(y2R)

            x1L = int(x1L)
            y1L = int(y1L)
            x2L = int(x2L)
            y2L = int(y2L)

            cv2.rectangle(frame_right, (x1R, y1R), (x2R, y2R), (0, 0, 255), 2)
            cv2.rectangle(frame_left, (x1L, y1L), (x2L, y2L), (0, 0, 255), 2)

            textR = "ID: {}".format(objectId_R)
            textL = "ID: {}".format(objectId_L)

            cv2.putText(frame_right, textR, (x1R, y1R-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            cv2.putText(frame_left, textL, (x1L, y1L-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)


        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame_right, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(frame_left, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Frame Right", frame_right)
        cv2.imshow("Frame Left", frame_left)


        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()
