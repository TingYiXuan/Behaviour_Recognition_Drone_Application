# import appdirs
import os

# Streaming settings
WIDTH, HEIGHT = 1200, 800
SEQUENCE_LENGTH = 30
VIDEO_FPS = 5

# App settings
APP_NAME = "Action Recognition"
APP_AUTHOR = "MCS23"
RECORDING_PATH = 'Action Recognition Recordings'

# Model name
MODEL_PATH = "./clean_data.h5"
ACTIONS = {0: 'Running', 1: 'Punching', 2: 'Waving' , 3: 'Kicking', 4: 'Walking'}
MAL_ACTIONS = ['Punching', 'Kicking']



def get_recording_folder() -> str:
    path = os.path.join(RECORDING_PATH)
    os.makedirs(path, exist_ok=True)
    return path

def get_unique_filename(stream_type: str, extension: str) -> str:
    path = get_recording_folder()
    result = stream_type

    count = 1
    print(path +  '/' + result + extension)
    while os.path.exists(path +  '/' + result + extension):
        result = stream_type + "_" + str(count)
        count += 1
    return result + extension
