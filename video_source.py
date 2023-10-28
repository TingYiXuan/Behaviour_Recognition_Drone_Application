"""
This module contains the VideoSource base class, which is used by the 
main loop of the application to get the video frame.

This module also contains the Webcam, Drone and PreRecorded classes, which
are the different video sources used by the application.

Author: Lim Yun Feng, Ting Yi Xuan, Chua Sheen Wey
Last Edited: 28/10/2023

Components:
    - VideoSource: The base class for the video source.
    - Webcam: The video source for the webcam.
    - Drone: The video source for the drone.
    - PreRecorded: The video source for the pre-recorded video.

Note: PreRecorded is not used in the application but only for testing the behaviour recognition model.
"""
import time
import cv2
import numpy as np
from djitellopy import Tello
from utils import HEIGHT, WIDTH


class VideoSource():
    """
    An abstract class used to represent the video source.

    Attributes:
        __has_frame (bool): Whether the video source has a frame.
    """
    def __init__(self) -> None:
        """
        The constructor for the VideoSource class.
        """
        self.__has_frame = False

    def start(self) -> None:
        """
        Callback function to start the video source. Executed prior to getting 
        the first frame but after starting the thread.
        """
        raise NotImplementedError

    def get_display_frame(self) -> tuple[bool, np.ndarray]:
        """
        Returns the current frame to be displayed on the screen.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing whether the frame is valid and the frame to be displayed.
        """
        raise NotImplementedError

    def next_frame(self) -> tuple[bool, np.ndarray]:
        """
        Callback function to get the next frame from the video source.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing whether the frame is valid and the frame to be displayed.
        """
        raise NotImplementedError
    
    def has_frame(self) -> bool:
        """
        Returns whether the video source has a frame.

        Returns:
            bool: Whether the video source has a frame.
        """
        return self.__has_frame
    
    def exit(self) -> None:
        """
        Callback function to exit the video source. Executed after the main loop
        has exited.
        """
        raise NotImplementedError
    
    def get_height(self) -> int:
        """
        Returns the height of the frame.

        Returns:
            int: The height of the video source.
        """
        raise NotImplementedError
    
    def get_width(self) -> int:
        """
        Returns the width of the frame. 

        Returns:
            int: The width of the video source.
        """
        raise NotImplementedError
    

class Webcam(VideoSource):
    """
    A class used to represent the webcam video source.

    Attributes:
        __cap (cv2.VideoCapture): The video capture object.
        __frame (numpy.ndarray): The current frame.
    """

    def __init__(self) -> None:
        """
        The constructor for the Webcam class.
        """
        super().__init__()
        self.__frame = None
        self.__has_frame = False

    def start(self) -> None:
        """
        Callback function to start the webcam video source.
        """
        self.__cap = cv2.VideoCapture(0)

        # Set the frame size
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    def get_display_frame(self) -> tuple[bool, np.ndarray]:
        """
        Returns the current frame to be displayed on the screen.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing whether the frame is valid and the frame to be displayed.
        """
        return self.__has_frame, self.__frame
    
    def next_frame(self) -> tuple[bool, np.ndarray]:
        """
        Callback function to get the next frame from the webcam video source.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing whether the frame is valid and the frame to be displayed.
        """
        self.__has_frame, self.__frame = self.__cap.read()
        return self.__has_frame, self.__frame
    
    def exit(self) -> None:
        """
        Callback function to exit the webcam video source.
        """
        self.__cap.release()

    def get_height(self) -> int:
        """
        Returns the height of the frame.

        Returns:
            int: The height of the webcam video source.
        """
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_width(self) -> int:
        """
        Returns the width of the frame.

        Returns:    
            int: The width of the webcam video source.
        """
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
class Drone:
    """
    A class used to represent the drone video source.

    Attributes:
        __drone (Tello): The drone object.
        __frame (numpy.ndarray): The current frame.
    """

    def __init__(self) -> None:
        """
        The constructor for the Drone class.
        """
        super().__init__()
        self.__drone = None
        self.__frame = None
        self.__has_frame = False

    def start(self) -> None:
        """
        Callback function to start the drone video source.

        Start the drone and takeoff.
        """
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
        """
        Returns the current frame to be displayed on the screen.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing whether the frame is valid and the frame to be displayed.
        """
        return self.__has_frame, cv2.cvtColor(self.__frame, cv2.COLOR_BGR2RGB)
    
    def next_frame(self) -> tuple[bool, np.ndarray]:
        """
        Callback function to get the next frame from the drone video source.

        Send the drone a command to confirm that the drone is still connected.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing whether the frame is valid and the frame to be displayed.
        """

        self.__drone.send_rc_control(0, 0, 0, 0)
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
        """
        Callback function to exit the drone video source.

        Land the drone and stop the video stream.
        """
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
    """
    A class used to represent the pre-recorded video source.

    Attributes:
        __cap (cv2.VideoCapture): The video capture object.
        __frame (numpy.ndarray): The current frame.
    """
    def __init__(self, filename: str) -> None:
        """
        The constructor for the PreRecorded class.

        Parameters:
            filename (str): The filename of the pre-recorded video.
        """
        super().__init__()
        self.__frame = None
        self.__has_frame = False
        self.__filename = filename

    def start(self) -> None:
        """
        Callback function to start the pre-recorded video source.
        """
        self.__cap = cv2.VideoCapture(self.__filename)

    def get_display_frame(self) -> tuple[bool, np.ndarray]:
        """
        Returns the current frame to be displayed on the screen.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing whether the frame is valid and the frame to be displayed.
        """
        return self.__has_frame, self.__frame
    
    def next_frame(self) -> tuple[bool, np.ndarray]:
        """
        Callback function to get the next frame from the pre-recorded video source.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing whether the frame is valid and the frame to be displayed.
        """
        self.__has_frame, self.__frame = self.__cap.read()
        return self.__has_frame, self.__frame
    
    def exit(self) -> None:
        """
        Callback function to exit the pre-recorded video source.
        """
        self.__cap.release()

    def get_height(self) -> int:
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_width(self) -> int:
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    