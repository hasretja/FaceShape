import cv2 as cv
import os
from time import localtime, strftime

class Camera(object):
    basedir = os.path.abspath(os.path.dirname(__file__))
    CAPTURES_DIR = basedir+"/static/captures/"
    RESIZE_RATIO = 1.0

    def __init__(self):
        self.video = cv.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        try:
            success, frame = self.video.read()
            if not success:
                return None
        except:
            print("çarptı")
            frame=cv.imread(Camera.basedir+r"\static\captures\28-11-2020-22h34m07s.jpg")


        if (Camera.RESIZE_RATIO != 1):
            frame = cv.resize(frame, None, fx=Camera.RESIZE_RATIO, \
                fy=Camera.RESIZE_RATIO)    
        return frame

    def get_feed(self):
        frame = self.get_frame()
        if frame is not None:
            ret, jpeg = cv.imencode('.jpg', frame)
            return jpeg.tobytes()

    def capture(self):
        frame = self.get_frame()
        timestamp = strftime("%d-%m-%Y-%Hh%Mm%Ss", localtime())
        filename = Camera.CAPTURES_DIR + timestamp +".jpg"
        if not cv.imwrite(filename, frame):
            print("resim çekilemedi")
            return "Unable to capture image"
            #raise RuntimeError("Unable to capture image "+timestamp)
        return timestamp
