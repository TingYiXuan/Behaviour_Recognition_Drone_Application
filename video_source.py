import time
import cv2
import numpy as np
from inference import get_recorder
from djitellopy import Tello

from utils import HEIGHT, WIDTH


class VideoSource():
    def __init__(self) -> None:
        self.__has_frame = False

    def start(self) -> None:
        raise NotImplementedError

    def get_display_frame(self) -> tuple[bool, np.ndarray]:
        raise NotImplementedError

    def next_frame(self) -> None:
        raise NotImplementedError
    
    def has_frame(self) -> bool:
        return self.__has_frame
    
    def exit(self) -> None:
        raise NotImplementedError
    
    def get_height(self) -> int:
        raise NotImplementedError
    
    def get_width(self) -> int:
        raise NotImplementedError
    

class Webcam(VideoSource):
    def __init__(self) -> None:
        super().__init__()
        self.__frame = None
        self.__has_frame = False

    def start(self) -> None:
        self.__cap = cv2.VideoCapture(0)
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    def get_display_frame(self) -> tuple[bool, np.ndarray]:
        return self.__has_frame, self.__frame
    
    def next_frame(self) -> tuple[bool, np.ndarray]:
        self.__has_frame, self.__frame = self.__cap.read()
        return self.__has_frame, self.__frame
    
    def exit(self) -> None:
        self.__cap.release()

    def get_height(self) -> int:
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_width(self) -> int:
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
class Drone:

    def __init__(self) -> None:
        super().__init__()
        self.__drone = None
        self.__frame = None
        self.__has_frame = False

    def start(self) -> None:
        # Connect to tello
        myDrone = Tello()
        myDrone.connect()
        myDrone.for_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0
        myDrone.yaw_velocity = 0
        myDrone.speed = 0
        print("Drone Battery:",  myDrone.get_battery())

        myDrone.streamon()
        time.sleep(5)
        myDrone.send_rc_control(0, 0, 0, 0)
        myDrone.takeoff()
        myDrone.send_rc_control(0, 0, 60, 0)
        time.sleep(2)
        myDrone.send_rc_control(0, 0, 0, 0)
        
        self.__drone = myDrone

    def get_display_frame(self) -> tuple[bool, np.ndarray]:
        return self.__has_frame, cv2.cvtColor(self.__frame, cv2.COLOR_BGR2RGB)
    
    def next_frame(self) -> tuple[bool, np.ndarray]:
        try:
            myFrame = self.__drone.get_frame_read()
        except:
            print("Drone Disconnected")
            self.__has_frame = False
            return self.__has_frame, None
        
        myFrame = myFrame.frame
        self.__frame = cv2.resize(myFrame, (WIDTH, HEIGHT))
        self.__has_frame = True
        return self.__has_frame, self.__frame

    def exit(self) -> None:
        try:    
            self.__drone.land()
        except:
            print("Drone Landed")
            
        self.__drone.streamoff()
    
    def get_height(self) -> int:
        return HEIGHT
    
    def get_width(self) -> int:
        return WIDTH
    
class PreRecorded(VideoSource):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.__frame = None
        self.__has_frame = False
        self.__filename = filename

    def start(self) -> None:
        self.__cap = cv2.VideoCapture(self.__filename)
        # self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        # self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    def get_display_frame(self) -> tuple[bool, np.ndarray]:
        return self.__has_frame, self.__frame
    
    def next_frame(self) -> tuple[bool, np.ndarray]:
        self.__has_frame, self.__frame = self.__cap.read()
        return self.__has_frame, self.__frame
    
    def exit(self) -> None:
        self.__cap.release()

    def get_height(self) -> int:
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_width(self) -> int:
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    