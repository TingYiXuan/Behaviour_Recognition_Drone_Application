import time
from PyQt5.QtCore import pyqtSignal, QObject, QRunnable
from PyQt5 import QtCore
from inference import *
from video_source import VideoSource

class Signals(QObject):
    """	
    Defines the signals available from a running worker thread.
    """
    start = pyqtSignal(int, name="start")
    complete = pyqtSignal(name="complete")
    alert = pyqtSignal(int, str, name="alert")

class StreamWorker(QRunnable):
    """
    Capture IP camera frames worker.
    """
    # def __init__(self, stream_function: callable, record: bool = False, filename: str = None, alert: bool = False):
    def __init__(self, source: VideoSource, record: bool = False, filename: str = None, alert: bool = False):
        super(StreamWorker, self).__init__()
        self.signals = Signals()

        # Private attributes
        self.__filename = filename
        self.__video_source = source
        self.__record = record
        self.__alert = alert
    
    @QtCore.pyqtSlot()
    def run(self) -> None:
        self.signals.start.emit(0)

        # Initialize the detector
        detector = PoseDetector()
        pTime = 0

        if self.__alert:
            model = YOLO(OBJECT_TRACKING_MODEL_PATH)
            mal_people = []

        self.__video_source.start() 

        if self.__record:
            self.__recorder = get_recorder(self.__filename, self.__video_source.get_width(), self.__video_source.get_height())
        else:
            self.__recorder = None

        while True:
            has_frame, frame = self.__video_source.next_frame()

            if not has_frame:
                raise ValueError("No frame captured")
            
            if self.__alert:
                track_result, action_result = tracking_inference(frame, model, detector)
            else:
                action_result = non_tracking_inference(frame, detector)


            has_frame, frame = self.__video_source.get_display_frame()

            if self.__alert and track_result is not None:
                draw_track_result(frame, track_result)

                if action_result.action in MAL_ACTIONS and track_result.id not in mal_people:
                    mal_people.append(track_result.id)
                    self.signals.alert.emit(track_result.id, action_result.action)

            if action_result is not None:
                draw_action_results(frame, action_result.action, action_result.probability)
            
            # Calculate FPS
            cTime = time.time()
            fps = 1 // (cTime - pTime)
            pTime = cTime
            draw_text(frame, fps, self.__video_source.get_height())
            
            if self.__recorder is not None:
                self.__recorder.write(frame)
            
            cv2.imshow("Frame", frame)

            # Event handler
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

        if self.__recorder is not None:
            self.__recorder.release()

        self.__video_source.exit()
        cv2.destroyAllWindows()

        self.signals.complete.emit()
     