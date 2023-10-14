import time
from ultralytics import YOLO
from inference import *
from pose_detector import PoseDetector
from utils import *
from video_source import PreRecorded, VideoSource, Webcam


def stream(video_source: VideoSource, record: bool = False, filename: str = None):
    results_history = []

    # Initialize the detector
    detector = PoseDetector()
    pTime = 0

    video_source.start() 

    fps_list = []

    if record:
        recorder = get_recorder(filename, video_source.get_width(), video_source.get_height())
    else:
        recorder = None

    while True:
        has_frame, frame = video_source.next_frame()
        if not has_frame:
            print("ERROR: No frame captured")
            break 

        
        action_result = non_tracking_inference(frame, detector)

        has_frame, frame = video_source.get_display_frame()

        results_history.append(action_result)

        if action_result is not None:
            draw_action_results(frame, action_result.action, action_result.probability)
        
        # Calculate FPS
        cTime = time.time()
        fps = 1 // (cTime - pTime)
        pTime = cTime
        fps_list.append(fps)
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
    
    # Find in history the most common action name of the tracked person
    action_history = {"Kicking": 0, "Punching": 0, "Running": 0, "Waving": 0, "Walking": 0}
    for action_result in results_history:
        if action_result is not None:
            action_history[action_result.action] += action_result.probability

    max_action = max(action_history, key=action_history.get)
    print("Most common action:", max_action)
    
    if action_history[max_action] == 0:
        max_action = None
    return sum(fps_list) / len(fps_list), max_action


def test_action_recognition():
    f = open("test_result_logs/test_action_recognition.txt", "w")
    f.write("3.2.3 Test Action Recognition\n")
    f.write("__________________________\n\n")

    test_names = ["'Kicking'", "'Walking'", "'Running'", "'Waving'", "'Punching'", "Low FPS", "High FPS"]
    test_videos = ["kicking.avi", "walking.avi", "running.avi", "waving.avi", "punching.avi", "low_fps_walking.mp4", "high_fps_running.mp4"]
    expected_outcomes = ["Kicking", "Walking", "Running", "Waving", "Punching", "Walking", "Running"]

    for i in range(len(test_names)):
        test_code = "3.2.3." + str(i + 1)
        f.write("Test {}: {} Action Recognition\n".format(test_code, test_names[i]))
        try:
            fps_result, action = stream(PreRecorded("assets/test_3_2_3/" + test_videos[i]), record=True, filename=test_videos[i])
            f.write("FPS: {}\n".format(fps_result))
        except Exception as e:
            f.write("ERROR: {}\n".format(e))
        else:
            if expected_outcomes[i] == action:
                f.write("Test {}: PASSED ({})\n".format(test_code, action))
            else:
                f.write("Test {}: FAILED (Expected: {}, Actual: {})\n".format(test_code, expected_outcomes[i], action))
        f.write("\n\n")

if __name__ == "__main__":
    test_action_recognition()