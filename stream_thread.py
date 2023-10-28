
"""
This module contains the StreamWorker class, which is a worker thread that captures frames from a video source and performs inference on them.

Author: Lim Yun Feng, Ting Yi Xuan, Chua Sheen Wey
Last Edited: 28/10/2023

Components:
    - StreamWorker: The worker thread that captures frames from a video source and performs inference on them.
"""
import time
from PyQt5.QtCore import pyqtSignal, QObject, QRunnable
from PyQt5 import QtCore
from inference import *
from video_source import VideoSource

class Signals(QObject):
    """	
    Defines the signals available from a running worker thread.

    Attributes:
        start (pyqtSignal): Signal emitted when the worker starts.
        complete (pyqtSignal): Signal emitted when the worker completes.
        alert (pyqtSignal): Signal emitted when the worker detects a malicious action.
    """
    start = pyqtSignal(int, name="start")
    complete = pyqtSignal(name="complete")
    alert = pyqtSignal(int, str, name="alert")

class StreamWorker(QRunnable):
    """
    Capture IP camera frames worker.

    Attributes:
        signals (Signals): The signals available from a running worker thread.
        __filename (str): The filename to save the recorded video.
        __video_source (VideoSource): The video source to capture frames from.
        __record (bool): Whether to record the video.
        __alert (bool): Whether to alert when a malicious action is detected.
        __recorder (cv2.VideoWriter): The video recorder.

    """
   
    def __init__(self, source: VideoSource, record: bool = False, filename: str = None, alert: bool = False):
        """
        The constructor for the StreamWorker class.

        Parameters:
            source (VideoSource): The video source to capture frames from.
            record (bool): Whether to record the video.
            filename (str): The filename to save the recorded video.
            alert (bool): Whether to alert when a malicious action is detected.
        """
        super(StreamWorker, self).__init__()
        self.signals = Signals()

        # Private attributes
        self.__filename = filename
        self.__video_source = source
        self.__record = record
        self.__alert = alert
    
    @QtCore.pyqtSlot()
    def run(self) -> None:
        """
        The main function of the StreamWorker class.

        Contains the main loop of the worker thread used to capture frames from a video source and perform inference on them.
        """
        # Emit start signal
        self.signals.start.emit(0)

        # Initialize the detector
        detector = PoseDetector()
        pTime = 0

        # Initialize the human-tracking model if alert is enabled
        if self.__alert:
            model = YOLO(OBJECT_TRACKING_MODEL_PATH)
            mal_people = []

        self.__video_source.start() 

        # Initialize the video recorder if recording is enabled
        if self.__record:
            self.__recorder = get_recorder(self.__filename, self.__video_source.get_width(), self.__video_source.get_height())
        else:
            self.__recorder = None

        # Main loop
        while True:
            # Get frame from video source
            has_frame, frame = self.__video_source.next_frame()
            if not has_frame:
                raise ValueError("No frame captured")
            
            # Perform inference based on whether alert is enabled
            if self.__alert:
                track_result, action_result = tracking_inference(frame, model, detector)
            else:
                action_result = non_tracking_inference(frame, detector)

            # get the original frame from video source
            has_frame, frame = self.__video_source.get_display_frame()

            # If alert is enabled, draw output of human-tracking model and alert if a malicious action is detected
            if self.__alert and track_result is not None:
                draw_track_result(frame, track_result)

                if action_result.action in MAL_ACTIONS and track_result.id not in mal_people:
                    mal_people.append(track_result.id)
                    self.signals.alert.emit(track_result.id, action_result.action)

            # display the output of action recognition model
            if action_result is not None:
                draw_action_results(frame, action_result.action, action_result.probability)
            
            # Calculate FPS and display it
            cTime = time.time()
            fps = 1 // (cTime - pTime)
            pTime = cTime
            draw_text(frame, fps, self.__video_source.get_height())
            
            # Record the frame if recording is enabled
            if self.__recorder is not None:
                self.__recorder.write(frame)
            
            # Display the frame
            cv2.imshow("Frame", frame)

            # Event handler
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
                
        if self.__recorder is not None:
            self.__recorder.release()

        self.__video_source.exit()
        cv2.destroyAllWindows()

        self.signals.complete.emit()
     