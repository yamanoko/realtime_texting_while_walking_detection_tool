import cv2


class DetectHuman:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        (bounding_boxes, weights) = self.hog.detectMultiScale(frame,
                                                              winStride=(16, 16),
                                                              padding=(4, 4),
                                                              scale=1)
        return bounding_boxes
