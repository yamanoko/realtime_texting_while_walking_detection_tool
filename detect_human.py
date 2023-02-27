import cv2
import tensorflow_hub as hub
import tensorflow as tf
"""
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
"""


class DetectHuman:
    def __init__(self):
        self.detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")

    def detect(self, frame):
        frame_tensor = tf.expand_dims(tf.constant(frame, dtype=tf.uint8), 0)
        detection_output = self.detector(frame_tensor)

        scores = detection_output['detection_scores'][0]
        boxes = detection_output['detection_boxes'][0]
        classes = detection_output['detection_classes'][0]

        qualified_boxes = boxes[(scores >= 0.8) & (classes == 1.0)]
        qualified_boxes = qualified_boxes.numpy()
        return qualified_boxes
