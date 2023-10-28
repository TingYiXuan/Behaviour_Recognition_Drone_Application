"""
This module contains the PoseDetector class, which is 
used to detect the pose of a person in a video frame.

It also displays the pose landmarks on the video frame.

Author: Lim Yun Feng, Ting Yi Xuan, Chua Sheen Wey
Last Edited: 28/10/2023

Components:
    - findPose: Detects the pose of a person in a video frame.
    - extract_keypoints: Extracts the pose keypoints from the video frame.
"""
import mediapipe as mp
import numpy as np
import tensorflow as tf
import cv2

from utils import MODEL_PATH

class PoseDetector:
    """
    A class used to detect the pose of a person in a video frame.

    Attributes:
        mode (bool): Whether to detect the pose in static image or video.
        upBody (bool): Whether to detect the upper body pose.
        smooth (bool): Whether to smooth the pose landmarks.
        detectionCon (float): Minimum confidence value for the pose detection to be considered successful.
        trackCon (float): Minimum confidence value for the pose tracking to be considered successful.
        mpDraw (mediapipe.solutions.drawing_utils): Used to draw the pose landmarks on the video frame.
        mpHolistic (mediapipe.solutions.holistic): Used to detect the pose of a person in a video frame.
        holistic (mediapipe.solutions.holistic.Holistic): The holistic model.
        drawSpec (mediapipe.solutions.drawing_utils.DrawingSpec): The drawing specification for the pose landmarks.
        new_model (tensorflow.python.keras.engine.sequential.Sequential): The sequential model used to predict the pose.
        sequence (list): The list of pose keypoints extracted from the video.
        sentence (list): The list of predicted pose labels.
    """

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        """
        The constructor for the PoseDetector class.

        Parameters:
            mode (bool): Whether to detect the pose in static image or video.
            upBody (bool): Whether to detect the upper body pose.
            smooth (bool): Whether to smooth the pose landmarks.
            detectionCon (float): Minimum confidence value for the pose detection to be considered successful.
            trackCon (float): Minimum confidence value for the pose tracking to be considered successful.
        """
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic(self.mode, self.upBody, self.smooth, False, False, False,
                                                 self.detectionCon, self.trackCon)
        self.drawSpec = self.mpDraw.DrawingSpec((255, 0, 0), thickness=1, circle_radius=1)

        self.new_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        self.sequence = []
        self.sentence = []

    def findPose(self, img, draw=True):
        """
        Detects the pose of a person in a video frame.

        Parameters:
            img (numpy.ndarray): The video frame.
            draw (bool): Whether to draw the pose landmarks on the video frame.

        Returns:
            results (mediapipe.python.solution_base.SolutionOutputs): The pose landmarks.
            img (numpy.ndarray): The video frame with the pose landmarks drawn on it.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(imgRGB)

        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.face_landmarks, self.mpHolistic.FACEMESH_CONTOURS,
                                           self.drawSpec, self.drawSpec)
                self.mpDraw.draw_landmarks(img, results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)
                self.mpDraw.draw_landmarks(img, results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS)

        return results, img

    def extract_keypoints(self, results):
        """
        Extracts the pose keypoints from the video frame.

        Parameters:
            results (mediapipe.python.solution_base.SolutionOutputs): The pose landmarks.

        Returns:
            np.concatenate([pose]) (numpy.ndarray): The pose keypoints.
        """
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        return np.concatenate([pose])