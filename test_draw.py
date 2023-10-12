from time import sleep
from cv2 import VideoCapture
import cv2

from inference import draw_text

SLEEP_TIME = 5

cap = VideoCapture(0)
frame = cap.read()[1]

def test_draw_text():

   draw_text(frame, 30, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
   cv2.imshow("Test with FPS 30", frame)
   cv2.waitKey(1)
   assert frame is not None

   sleep(SLEEP_TIME)
   cv2.destroyAllWindows()

   draw_text(frame, 50, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
   cv2.imshow("Test with FPS 50", frame)
   cv2.waitKey(1)
   assert frame is not None

   sleep(SLEEP_TIME)
   cv2.destroyAllWindows()
   

def test_end():
   cap.release()
   cv2.destroyAllWindows()