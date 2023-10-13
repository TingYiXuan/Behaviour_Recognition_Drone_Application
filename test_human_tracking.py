import time
from cv2 import VideoCapture
from ultralytics import YOLO
from inference import *
from pose_detector import PoseDetector
from utils import *
from video_source import PreRecorded, VideoSource

def stream(video_source: VideoSource, record: bool = False, filename: str = None):
    # Initialize the detector
    model = YOLO(OBJECT_TRACKING_MODEL_PATH)
    pTime = 0

    video_source.start() 

    if record:
        recorder = get_recorder(filename, video_source.get_width(), video_source.get_height())
    else:
        recorder = None

    while True:
        has_frame, frame = video_source.next_frame()
        if not has_frame:
            print("ERROR: No frame captured")
            break 

        tracking_result = track_person(frame, model)
        has_frame, frame = video_source.get_display_frame()

        if tracking_result is not None:
            draw_track_result(frame, tracking_result)

        # Calculate FPS
        cTime = time.time()
        fps = 1 // (cTime - pTime)
        pTime = cTime
        draw_text(frame, fps, video_source.get_height())

        if recorder is not None:
            recorder.write(frame)
        
        cv2.imshow("Frame", frame)

        # Event handler
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    if recorder is not None:
        recorder.release()

    video_source.exit()
    cv2.destroyAllWindows()

def one_frame(img_path: str):
    cap = VideoCapture(img_path)
    frame = cap.read()[1]

    model = YOLO(OBJECT_TRACKING_MODEL_PATH)
    result = track_person(frame, model)
    
    detector = PoseDetector()
    predict_pose(frame, detector)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    time.sleep(5)
    
    cap.release()
    cv2.destroyAllWindows()
    return result


def test_pose_estimation():
    f = open("test_result_logs/test_human_tracking.txt", "w")
    f.write("3.2.2 Test Human Tracking\n")
    f.write("__________________________\n\n")

    # Test 1
    test_code = "3.2.2.1"
    video_name = "basic_kicking.avi"
    f.write("Test {}: Basic pose estimation\n".format(test_code))
    try:
        stream(PreRecorded("assets/test_3_2_1/" + video_name), record=True, filename=video_name)
    except Exception as e:
        f.write("Test {}: ERROR ({})\n".format(test_code, e))
    else:
        f.write("Test {}: Completed without errors\n".format(test_code))
    f.write("\n\n")

    # Test 2
    test_code = "3.2.2.2"
    video_name = "low_light_kicking.avi"
    f.write("Test {}: Low light pose estimation\n".format(test_code))
    try:
        stream(PreRecorded("assets/test_3_2_1/" + video_name), record=True, filename=video_name)
    except Exception as e:
        f.write("Test {}: ERROR ({})\n".format(test_code, e))
    else:
        f.write("Test {}: Completed without errors\n".format(test_code))
    f.write("\n\n")


    # Test 3
    test_code = "3.2.2.3"
    video_name = "high_fps_kicking.mp4"
    f.write("Test {}: High FPS pose estimation\n".format(test_code))
    try:
        stream(PreRecorded("assets/test_3_2_1/" + video_name), record=True, filename=video_name)
    except Exception as e:
        f.write("Test {}: ERROR ({})\n".format(test_code, e))
    else:
        f.write("Test {}: Completed without errors\n".format(test_code))
    f.write("\n\n")


    # Test 4
    test_code = "3.2.2.4"
    video_name = "low_fps_kicking.mp4"
    f.write("Test {}: Low FPS pose estimation\n".format(test_code))
    try:
        stream(PreRecorded("assets/test_3_2_1/" + video_name), record=True, filename=video_name)
    except Exception as e:
        f.write("Test {}: ERROR ({})\n".format(test_code, e))
    else:
        f.write("Test {}: Completed without errors\n".format(test_code))
    f.write("\n\n")

    # Test 5
    test_code = "3.2.2.5"
    video_name = "no_human.mp4"
    f.write("Test {}: Multiple Human pose estimation\n".format(test_code))
    try:
        stream(PreRecorded("assets/test_3_2_1/" + video_name), record=True, filename=video_name)
    except Exception as e:
        f.write("Test {}: ERROR ({})\n".format(test_code, e))
    else:
        f.write("Test {}: Completed without errors\n".format(test_code))
    f.write("\n\n")

    # Test 6
    test_code = "3.2.2.6"
    video_name = "multiple_human.mp4"
    f.write("Test {}: Multiple Human pose estimation\n".format(test_code))
    try:
        stream(PreRecorded("assets/test_3_2_1/" + video_name), record=True, filename=video_name)
    except Exception as e:
        f.write("Test {}: ERROR ({})\n".format(test_code, e))
    else:
        f.write("Test {}: Completed without errors\n".format(test_code))
    f.write("\n\n")

    # Test 7
    test_code = "3.2.2.7"
    video_name = "large_human.avi"
    f.write("Test {}: Large Scale pose estimation\n".format(test_code))
    try:
        stream(PreRecorded("assets/test_3_2_2/" + video_name), record=True, filename=video_name)
    except Exception as e:
        f.write("Test {}: ERROR ({})\n".format(test_code, e))
    else:
        f.write("Test {}: Completed without errors\n".format(test_code))
    f.write("\n\n")

    # Test 8
    test_code = "3.2.2.8"
    video_name = "small_human.avi"
    f.write("Test {}: Small scale pose estimation\n".format(test_code))
    try:
        stream(PreRecorded("assets/test_3_2_2/" + video_name), record=True, filename=video_name)
    except Exception as e:
        f.write("Test {}: ERROR ({})\n".format(test_code, e))
    else:
        f.write("Test {}: Completed without errors\n".format(test_code))
    f.write("\n\n")

    # Test 9
    test_code = "3.2.2.9"
    video_name = "top_down_kicking.avi"
    f.write("Test {}: Top Down pose estimation\n".format(test_code))
    try:
        stream(PreRecorded("assets/test_3_2_1/" + video_name), record=True, filename=video_name)
    except Exception as e:
        f.write("Test {}: ERROR ({})\n".format(test_code, e))
    else:
        f.write("Test {}: Completed without errors\n".format(test_code))
    f.write("\n\n")



    f.close()

if __name__ == '__main__':
    test_pose_estimation()