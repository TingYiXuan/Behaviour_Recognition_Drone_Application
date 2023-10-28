"""
This module contains functions and constants used by other modules in the system.

Author: Lim Yun Feng, Ting Yi Xuan, Chua Sheen Wey
Last Edited: 28/10/2023

Components:
    - MainWindow: The main window of the application
    - main: The main (funciton) entry point of the application
"""
import os

# Frame configurations
WIDTH, HEIGHT = 1200, 800
SEQUENCE_LENGTH = 30
VIDEO_FPS = 5

# App configurations
APP_NAME = "Action Recognition"
APP_AUTHOR = "MCS23"
RECORDING_PATH = 'behaviour_recognition_recordings'

# Model configurations
MODEL_PATH = "./lstm_action_recognition.h5"
OBJECT_TRACKING_MODEL_PATH = "yolov8n_human_tracking.pt"
ACTIONS = {0: 'Running', 1: 'Punching', 2: 'Waving' , 3: 'Kicking', 4: 'Walking'}
MAL_ACTIONS = ['Punching', 'Kicking']


def get_recording_folder() -> str:
    """
    Returns the path to the recording folder, create the directory if it does not exist.

    Returns:
        str: The path to the recording folder.
    """
    path = os.path.join(RECORDING_PATH)
    os.makedirs(path, exist_ok=True)
    return path

def get_unique_filename(stream_type: str, extension: str) -> str:
    """
    Returns a unique filename for the recording.

    Parameters:
        stream_type (str): The type of the stream.
        extension (str): The extension of the file.

    Returns:
        str: The unique filename for the recording.
    """
    path = get_recording_folder()
    result = stream_type

    count = 1
    print(path +  '/' + result + extension)
    while os.path.exists(path +  '/' + result + extension):
        result = stream_type + "_" + str(count)
        count += 1
    return result + extension
