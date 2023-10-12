import numpy as np
import cv2
import math

from ultralytics import YOLO
from pose_detector import PoseDetector
from utils import *

class ActionDetectorResult:
    def __init__(self, action: str, probability: float):
        self.action = action
        self.probability = probability

    def __str__(self):
        return f"Action: {self.action}, Probability: {self.probability}"

class TrackingResult:
    def __init__(self, x: int, y: int, w: int, h: int, id: int, roi):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = id
        self.roi = roi

    def __str__(self):
        return f"X: {self.x}, Y: {self.y}, W: {self.w}, H: {self.h}, ID: {self.id}"

def get_recorder(filename: str, width: int, height: int):
    if filename is None or filename[-4:] != ".avi":
        raise ValueError("Filename cannot be None if record is True")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        get_recording_folder() + '/' + filename, 
        fourcc, 
        fps=VIDEO_FPS, 
        frameSize=(width, int(height))
        )

    return out

def predict_pose(frame, detector: PoseDetector) -> tuple[any, list]:
    result, _ = detector.findPose(frame)

    # extract keypoints
    keypoints = detector.extract_keypoints(result)
    detector.sequence.append(keypoints)
    detector.sequence = detector.sequence[-SEQUENCE_LENGTH:]

    return keypoints, detector.sequence

def has_valid_pose(keypoints, sequence: list) -> bool:
    return sum(keypoints) != 0 and len(sequence) == SEQUENCE_LENGTH

def predict_action(detector: PoseDetector, sequence: list) -> tuple[str, float]:
    res = detector.new_model.predict(np.expand_dims(sequence, axis=0))[0]
    class_idx = int(np.argmax(res)) 
    return ACTIONS[class_idx], res[class_idx]

def draw_text(frame, fps: int, height: int):
    cv2.rectangle(frame, (0, 0), (400, 50), (245, 50, 16), -1)
    cv2.putText(frame, 
                'FPS: {}'.format(fps), 
                 (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    # Quit Notification text
    cv2.putText(frame, "Press 'Q' to Exit", (10, int(height) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def draw_action_results(frame, action: str, probability: float):
    cv2.rectangle(frame, (0, 0), (400, 130), (245, 50, 16), -1)
    cv2.putText(frame, 'Action: {}'.format(action), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(frame, 
                'Probability: {}'.format(math.floor(probability * 1000) / 1000), (20, 110), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

def track_person(frame, model: YOLO) -> TrackingResult | None:
    track_results = model.track(frame, persist=True, show=False, classes = 0)
    track_results = model.predict(frame)
    box_data = track_results[0].boxes.xywh

    if len(box_data) > 0:
        x, y, w, h, id = box_data[0][0], box_data[0][1], box_data[0][2], box_data[0][3], track_results[0].boxes.data[0][4]
        
        x = int(x - w/2)
        y = int(y - h/2)
        w = int(w)
        h = int(h)

        roi = frame[y:y+h, x:x+w]
        return TrackingResult(x, y, w, h, int(id), roi)
    else:
        return None

def draw_track_result(frame, results: TrackingResult):
    #  Plot the box
    cv2.putText(frame, 'ID: {}'.format(results.id), (results.x, results.y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (245, 50, 16), 2)
    cv2.rectangle(frame, (results.x, results.y), (results.x + results.w, results.y + results.h), (245, 50, 16), 2)

def non_tracking_inference(frame, detector: PoseDetector) -> ActionDetectorResult | None:
    keypoints, sequence = predict_pose(frame, detector)

    if has_valid_pose(keypoints, sequence):
        action, probability = predict_action(detector, sequence)
        return ActionDetectorResult(action, probability)
    else:
        return None
    
def tracking_inference(frame, model: YOLO, detector: PoseDetector) -> tuple[TrackingResult | None, ActionDetectorResult | None]:
    tracking_result = track_person(frame, model)
    if tracking_result is None:
        return None, None

    keypoints, sequence = predict_pose(tracking_result.roi, detector)

    if has_valid_pose(keypoints, sequence):
        action, probability = predict_action(detector, sequence)
        return tracking_result, ActionDetectorResult(action, probability)
    else:
        return None, None


