"""
This module contains the inference logic for the application.

Specifications of each module is implemented in this module

Author: Lim Yun Feng, Ting Yi Xuan, Chua Sheen Wey
Last Edited: 28/10/2023

Components:
    - get_recorder: Returns a cv2 video writer object.
    - predict_pose: Predicts the pose of a person in a video frame.
    - has_valid_pose: Checks whether the pose is valid.
    - predict_action: Predicts the action of a person in a video frame.
    - draw_text: Draws the text on the video frame.
    - draw_action_results: Draws the action results on the video frame.
    - track_person: Tracks the person in the video frame.
    - draw_track_result: Draws the tracking result on the video frame.
    - non_tracking_inference: Performs inference on the video frame without tracking.
    - tracking_inference: Performs inference on the video frame with tracking.
"""
import numpy as np
import cv2
import math
from ultralytics import YOLO
from pose_detector import PoseDetector
from utils import *

class ActionDetectorResult:
    """
    A class used to represent the result of the action detector.

    Attributes:
        action (str): The predicted action.
        probability (float): The probability of the predicted action.
    """
    def __init__(self, action: str, probability: float):
        """
        The constructor for the ActionDetectorResult class.

        Parameters:
            action (str): The predicted action.
            probability (float): The probability of the predicted action.
        """
        self.action = action
        self.probability = probability

    def __str__(self):
        """
        Returns the string representation of the ActionDetectorResult object.

        Returns:
            str: The string representation of the ActionDetectorResult object.
        """
        return f"Action: {self.action}, Probability: {self.probability}"

class TrackingResult:
    """
    A class used to represent the result of the human tracking.

    Attributes:
        x (int): The x-coordinate of the bounding box.
        y (int): The y-coordinate of the bounding box.
        w (int): The width of the bounding box.
        h (int): The height of the boundingassigned to the person.
        roi (numpy.ndarray): The region of interest of the bounding box.
    """
    def __init__(self, x: int, y: int, w: int, h: int, id: int, roi):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = id
        self.roi = roi

    def __str__(self):
        """
        Returns the string representation of the TrackingResult object.

        Returns:
            str: The string representation of the TrackingResult object.
        """
        return f"X: {self.x}, Y: {self.y}, W: {self.w}, H: {self.h}, ID: {self.id}"

def get_recorder(filename: str, width: int, height: int):
    """
    Returns a cv2 video writer object.

    Parameters:
        filename (str): The filename of the video.
        width (int): The width of the video.
        height (int): The height of the video.

    Returns:
        cv2.VideoWriter: The video writer object.
    """
    # check to see if the filename is None
    if filename is None:
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
    """
    Predicts the pose of a person in a video frame. 

    Parameters:
        frame (numpy.ndarray): The video frame.
        detector (PoseDetector): The pose detector.

    Returns:
        tuple[any, list]: The keypoints and the sequence of keypoints.
    """
    result, _ = detector.findPose(frame)

    # extract keypoints
    keypoints = detector.extract_keypoints(result)
    detector.sequence.append(keypoints)
    detector.sequence = detector.sequence[-SEQUENCE_LENGTH:]

    return keypoints, detector.sequence

def has_valid_pose(keypoints, sequence: list) -> bool:
    """
    Checks whether the pose is valid.

    Parameters:
        keypoints (any): The keypoints.
        sequence (list): The sequence of keypoints.

    Returns:
        bool: Whether the pose is valid.
    """
    return sum(keypoints) != 0 and len(sequence) == SEQUENCE_LENGTH

def predict_action(detector: PoseDetector, sequence: list) -> tuple[str, float]:
    """
    Predicts the action of a person in a video frame.

    Parameters:
        detector (PoseDetector): The pose detector.
        sequence (list): The sequence of keypoints.

    Returns:
        tuple[str, float]: The predicted action and the probability of the predicted action.
    """
    res = detector.new_model.predict(np.expand_dims(sequence, axis=0))[0]
    class_idx = int(np.argmax(res)) 
    return ACTIONS[class_idx], res[class_idx]

def draw_text(frame, fps: int, height: int):
    """
    Draws any text that are present in each frame.

    Parameters:
        frame (numpy.ndarray): The video frame.
        fps (int): The FPS of the video.
        height (int): The height of the video.
    """
    # FPS text
    cv2.rectangle(frame, (0, 0), (400, 50), (245, 50, 16), -1)
    cv2.putText(frame, 
                'FPS: {}'.format(fps), 
                 (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    # Quit Notification text
    cv2.putText(frame, "Press 'Q' to Exit", (10, int(height) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def draw_action_results(frame, action: str, probability: float):
    """
    Draws the action results on the video frame.

    Parameters:
        frame (numpy.ndarray): The video frame.
        action (str): The predicted action.
        probability (float): The probability of the predicted action.
    """
    cv2.rectangle(frame, (0, 0), (400, 130), (245, 50, 16), -1)
    cv2.putText(frame, 'Action: {}'.format(action), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(frame, 
                'Probability: {}'.format(math.floor(probability * 1000) / 1000), (20, 110), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

def track_person(frame, model: YOLO) -> TrackingResult | None:
    """
    Tracks the person in the video frame.

    Parameters:
        frame (numpy.ndarray): The video frame.
        model (YOLO): The YOLO model.

    Returns:
        TrackingResult | None: The tracking result if there is any.
    """
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
    """
    Draws the tracking result on the video frame.

    Parameters:
        frame (numpy.ndarray): The video frame.
        results (TrackingResult): The tracking result.
    """
    #  Plot the box
    cv2.putText(frame, 'ID: {}'.format(results.id), (results.x, results.y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (245, 50, 16), 2)
    cv2.rectangle(frame, (results.x, results.y), (results.x + results.w, results.y + results.h), (245, 50, 16), 2)

def non_tracking_inference(frame, detector: PoseDetector) -> ActionDetectorResult | None:
    """
    Performs inference on the video frame without tracking. Only performs pose estimation and action recognition

    Parameters:
        frame (numpy.ndarray): The video frame.
        detector (PoseDetector): The pose detector.

    Returns:
        ActionDetectorResult | None: The action detector result if there is any.
    """
    keypoints, sequence = predict_pose(frame, detector)

    if has_valid_pose(keypoints, sequence):
        action, probability = predict_action(detector, sequence)
        return ActionDetectorResult(action, probability)
    else:
        return None
    
def tracking_inference(frame, model: YOLO, detector: PoseDetector) -> tuple[TrackingResult | None, ActionDetectorResult | None]:
    """
    Performs inference on the video frame with tracking. Performs human tracking, pose estimation, action recognition in this order.

    Parameters:
        frame (numpy.ndarray): The video frame.
        model (YOLO): The YOLO model.
        detector (PoseDetector): The pose detector.

    Returns:
        tuple[TrackingResult | None, ActionDetectorResult | None]: The tracking result and the action detector result if there is any.
    """
    tracking_result = track_person(frame, model)
    if tracking_result is None:
        return None, None

    keypoints, sequence = predict_pose(tracking_result.roi, detector)

    if has_valid_pose(keypoints, sequence):
        action, probability = predict_action(detector, sequence)
        return tracking_result, ActionDetectorResult(action, probability)
    else:
        return None, None


