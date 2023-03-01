import numpy as np
import tensorflow as tf
from predict import Predict


class DetectHuman(Predict):
    def __init__(self, frame_width, frame_height, model_path='ssd_mobilenet_v1_1_metadata_1.tflite'):
        super().__init__(model_path)

        self.model_input_size = [300, 300]

        self.width_ratio = frame_width / self.model_input_size[0]
        self.height_ratio = frame_height / self.model_input_size[1]

        self.ratio_array = np.array([self.height_ratio, self.width_ratio, self.height_ratio, self.width_ratio])

    def detect(self, frame):
        resized_frame = tf.image.resize(frame, self.model_input_size)
        resized_frame = tf.cast(resized_frame, dtype=tf.uint8)
        resized_frame = tf.expand_dims(resized_frame, 0)

        prediction = super().predict(resized_frame)
        boxes = prediction[0][0]
        classes = prediction[0][1]
        scores = prediction[0][2]

        qualified_boxes = boxes[(scores >= 0.8) & (classes == 1)]

        if np.size(qualified_boxes) == 0:
            return None

        qualified_boxes *= self.ratio_array

        return qualified_boxes
