"""
Node template for creating custom nodes.
"""

from typing import Any, Dict
import cv2
#playsound version is 1.2.2
from playsound import playsound
import threading
import tensorflow as tf
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


def map_bbox_to_image_coords(img_array, box):
    img_height = img_array.shape[0]
    img_width = img_array.shape[1]

    x_max = int(box[2] * img_width)
    x_min = int(box[0] * img_width)
    y_max = int(box[3] * img_height)
    y_min = int(box[1] * img_height)

    return x_max, x_min, y_max, y_min


def slice_img(img_array, box):
    x_max, x_min, y_max, y_min = map_bbox_to_image_coords(img_array, box)
    sliced_img_array = img_array[y_min:y_max, x_min:x_max]
    return sliced_img_array


def resize_img(sliced_img_array):
    sliced_img_tensor = tf.constant(sliced_img_array)
    sliced_img_tensor = tf.expand_dims(sliced_img_tensor, 0)
    resized_img_tensor = tf.image.resize_with_pad(sliced_img_tensor, 256, 256)

    return resized_img_tensor


def predict_img_percentage(img_tensor_expanded, model):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], img_tensor_expanded)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])
    prediction_value = prediction[0][0]
    distracted_percentage = 100 * (1 - prediction_value)
    distracted_percentage = round(distracted_percentage, 2)
    return distracted_percentage


def put_text(percentage, img, org):
    str_img_percentage = str(percentage)
    cv2.putText(
        img=img,
        text=str_img_percentage,
        org=org,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(0, 0, 255),
        thickness=3,
    )


def sound():
    playsound('caution.mp3')


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")
        self.frame_count = 0
        self.sound_task = threading.Thread()

        model_path = 'lite_model_2.tflite'
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()

    def sound_play(self):
        if self.sound_task.is_alive():
            return

        self.sound_task = threading.Thread(target=sound)
        self.sound_task.start()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        self.frame_count += 1
        if self.frame_count >= 1000:
            self.frame_count = 0

        if (self.frame_count % 5) != 0:
            return {}

        bboxes = inputs["bboxes"]
        if bboxes is None:
            return {}

        img = inputs["img"]

        for bbox in bboxes:
            x_max, x_min, y_max, y_min = map_bbox_to_image_coords(img, bbox)

            sliced_img = slice_img(img, bbox)
            resized_img_tensor = resize_img(sliced_img)
            img_percentage = predict_img_percentage(resized_img_tensor, self.model)

            if img_percentage >= 80:
                #put_text(percentage=img_percentage, img=img, org=(x_min, y_min))
                self.sound_play()
                break

        return {}
