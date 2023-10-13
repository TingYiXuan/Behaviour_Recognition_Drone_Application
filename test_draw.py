from time import sleep
from cv2 import VideoCapture
import cv2

from inference import draw_action_results, draw_text, predict_pose
from pose_detector import PoseDetector

import unittest


SLEEP_TIME = 5
BLANK_IMG_FILE = "assets/black.jpg"
HUMAN_IMG_FILE = "assets/human.jpg"
LIVING_ROOM_IMG_FILE = "assets/living_room.jpg"


class TestDraw(unittest.TestCase):
   
    def test_draw_text(self):
        print("\nTest draw text")
        test = [30, 50]

        for i, fps in enumerate(test):
            print("Test {}: with FPS".format(i), fps)
            cap = VideoCapture(BLANK_IMG_FILE)
            frame = cap.read()[1]

            draw_text(frame, 30, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cv2.imshow("Test with FPS" + str(fps), frame)
            cv2.waitKey(1)
            self.assertTrue(frame is not None    , "Frame is None")
            print("PASSED")

            sleep(SLEEP_TIME)
            cv2.destroyAllWindows()
            cap.release()

    def test_draw_pose(self):
        # Test with no  human
        
        print("\nTest draw pose")

        imgs = [LIVING_ROOM_IMG_FILE, HUMAN_IMG_FILE]
        tests = ["Test with no human", "Test with human"]

        for i, img in enumerate(imgs):
            print("Test ", str(i), ":", tests[i])
            cap = VideoCapture(img)
            frame = cap.read()[1]

            self.assertTrue(frame is not None    , "Frame is None")
            print("PASSED")

            detector = PoseDetector()
            predict_pose(frame, detector)

            cv2.imshow(tests[i], frame)
            cv2.waitKey(1)

            sleep(SLEEP_TIME)
            
            cap.release()
            cv2.destroyAllWindows()

    def test_draw_action(self):
        print("\nTest draw action")

        actions = ["Running", "Kicking"]
        probability = [0.7, 0.8]

        for i, action in enumerate(actions):
            print("Test" + str(i), ": Test with action {} with probability of {}".format(action, probability[i]))
            
            cap = VideoCapture(BLANK_IMG_FILE)
            frame = cap.read()[1]

            draw_action_results(frame, action, probability[i])
            cv2.imshow("Test with action {} with probability of {}".format(action, probability[i]), frame)
            cv2.waitKey(1)
            self.assertTrue(frame is not None, "Frame is None")
            print("PASSED")

            sleep(SLEEP_TIME)
            cv2.destroyAllWindows()
            cap.release()

if __name__ == '__main__':
    unittest.main()



