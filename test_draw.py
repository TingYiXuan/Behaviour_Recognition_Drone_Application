from time import sleep
from cv2 import VideoCapture
import cv2

from inference import draw_action_results, draw_text, predict_pose
from pose_detector import PoseDetector

SLEEP_TIME = 5
BLANK_IMG_FILE = "assets/black.jpg"
HUMAN_IMG_FILE = "assets/human.jpg"
LIVING_ROOM_IMG_FILE = "assets/living_room.jpg"


def test_draw_text():
   test = [30, 50]

   for fps in test:
      print("Test with FPS", fps)
      cap = VideoCapture(BLANK_IMG_FILE)
      frame = cap.read()[1]

      draw_text(frame, 30, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
      cv2.imshow("Test with FPS" + str(fps), frame)
      cv2.waitKey(1)
      assert (frame is not None) , "Frame is None"
      print("Test with FPS", fps, "passed")

      sleep(SLEEP_TIME)
      cv2.destroyAllWindows()
      cap.release()

def test_draw_pose():
   # Test with no human

   imgs = [LIVING_ROOM_IMG_FILE, HUMAN_IMG_FILE]
   tests = ["Test with no human", "Test with human"]

   for i, img in enumerate(imgs):
      print(tests[i])
      cap = VideoCapture(img)
      frame = cap.read()[1]

      assert (frame is not None), "Frame error"
      print(tests[i], "passed")

      detector = PoseDetector()
      predict_pose(frame, detector)

      cv2.imshow(tests[i], frame)
      cv2.waitKey(1)

      sleep(SLEEP_TIME)
      
      cap.release()
      cv2.destroyAllWindows()

def test_draw_action():
   actions = ["Running", "Kicking"]
   probability = [0.7, 0.8]

   for i, action in enumerate(actions):
      print("Test with action {} with probability of {}".format(action, probability[i]))
      
      cap = VideoCapture(BLANK_IMG_FILE)
      frame = cap.read()[1]

      draw_action_results(frame, action, probability[i])
      cv2.imshow("Test with action {} with probability of {}".format(action, probability[i]), frame)
      cv2.waitKey(1)
      assert (frame is not None) , "Frame is None"
      print("Test with action {} with probability of {} PASSED".format(action, probability[i]))

      sleep(SLEEP_TIME)
      cv2.destroyAllWindows()
      cap.release()




