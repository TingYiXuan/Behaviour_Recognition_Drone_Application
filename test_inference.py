import time
from ultralytics import YOLO
from inference import *
from pose_detector import PoseDetector
from utils import *
from video_source import PreRecorded, VideoSource, Webcam


def stream(video_source: VideoSource, record: bool = False, filename: str = None, alert: bool = False):
    alert_received = []
    results_history = []

    # Initialize the detector
    detector = PoseDetector()
    pTime = 0
    
    if alert:
        model = YOLO(OBJECT_TRACKING_MODEL_PATH)
        mal_people = []

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

        if alert:
            track_result, action_result = tracking_inference(frame, model, detector)
        else:
            action_result = non_tracking_inference(frame, detector)
            track_result = None


        has_frame, frame = video_source.get_display_frame()

        results_history.append((track_result, action_result))

        if alert and track_result is not None:
            draw_track_result(frame, track_result)

            if action_result.action in MAL_ACTIONS and track_result.id not in mal_people:
                mal_people.append(track_result.id)
                alert_received.append((track_result.id, action_result.action))

        if action_result is not None:
            draw_action_results(frame, action_result.action, action_result.probability)
        
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

    print("Alerts:", alert_received)

    if alert:
        # Find in history the id of the most commonly tracked person
        tracked_people = [0] * 200
        for track_result, _ in results_history:
            if track_result is not None:
                tracked_people[track_result.id] += 1
        
        max_id = tracked_people.index(max(tracked_people))
        print("Most appearance (ID):", max_id)
    
    # Find in history the most common action name of the tracked person
    action_history = {"Kicking": 0, "Punching": 0, "Running": 0, "Waving": 0, "Walking": 0}
    for track_result, action_result in results_history:
        if action_result is not None and (not alert or track_result.id == max_id):
            action_history[action_result.action] += action_result.probability

    max_action = max(action_history, key=action_history.get)
    print("Most common action:", max_action)
    
    if alert:
        return max_id, max_action
    else:
        return None, max_action

if __name__ == '__main__':
    has_alert = False
    # result1 = stream(PreRecorded("assets/basic_kicking_1.avi"), record=True, filename="basic_kicking_1.avi", alert=has_alert)
    # result2 = stream(PreRecorded("assets/basic_kicking_2.avi"), record=True, filename="basic_kicking_2.avi", alert=has_alert)
    # result3 = stream(PreRecorded("assets/basic_waving_1.avi"), record=True, filename="basic_waving_1.avi", alert=has_alert)
    # result4 = stream(PreRecorded("assets/basic_waving_2.avi"), record=True, filename="basic_waving_2.avi", alert=has_alert)
    # result5 = stream(PreRecorded("assets/basic_walking_1.avi"), record=True, filename="basic_walking_1.avi", alert=has_alert)
    # result6 = stream(PreRecorded("assets/basic_walking_2.avi"), record=True, filename="basic_walking_2.avi", alert=has_alert)
    # result7 = stream(PreRecorded("assets/basic_punching_1.avi"), record=True, filename="basic_punching_1.avi", alert=has_alert)
    # result8 = stream(PreRecorded("assets/basic_punching_2.avi"), record=True, filename="basic_punching_2.avi", alert=has_alert)
    # result9 = stream(PreRecorded("assets/basic_running_1.avi"), record=True, filename="basic_running_1.avi", alert=has_alert)
    # result10 = stream(PreRecorded("assets/basic_running_2.avi"), record=True, filename="basic_running_2.avi", alert=has_alert)
    
    result11 = stream(PreRecorded("assets/low_speed_running_2.mp4"), record=True, filename="low_speed_running_2.avi", alert=has_alert)
    # result12 = stream(PreRecorded("assets/speed_up_walking_2.mp4"), record=True, filename="speed_up_walking_2.avi", alert=has_alert)
    
    # result13 = stream(PreRecorded("assets/low_light_waving_1.avi"), record=True, filename="low_light_waving_1.avi", alert=has_alert)

    # print(result1)
    # print(result2)
    # print(result3)
    # print(result4)
    # print(result5)
    # print(result6)
    # print(result7)
    # print(result8)
    # print(result9)
    # print(result10)
    print(result11)
    # print(result12)
    # print(result13)