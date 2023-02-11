"""
Node template for creating custom nodes.
"""

from typing import Any, Dict
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from peekingduck.pipeline.nodes.abstract_node import AbstractNode
import pathlib


def map_bbox_to_image_coords(img_array, box):
    img_height = img_array.shape[0]
    img_width = img_array.shape[1]

    x_max = box[2] * img_width
    x_min = box[0] * img_width
    y_max = box[3] * img_height
    y_min = box[1] * img_height

    return x_max, x_min, y_max, y_min


def slice_img(img_array, box):
    x_max, x_min, y_max, y_min = map_bbox_to_image_coords(img_array, box)
    sliced_img_array = img_array[int(y_min):int(y_max), int(x_min):int(x_max)]
    return sliced_img_array


def resize_img(sliced_img_array):
    sliced_img_tensor = tf.constant(sliced_img_array)
    sliced_img_tensor = tf.expand_dims(sliced_img_tensor, 0)
    resized_img_tensor = tf.image.resize_with_pad(sliced_img_tensor, 256, 256)

    return resized_img_tensor


def predict_img_percentage(img_tensor_expanded, model):
    prediction = model.predict(img_tensor_expanded)
    prediction = prediction[0]
    prediction_value = prediction[0]
    distracted_percentage = 100 * (1 - prediction_value)
    distracted_percentage = round(distracted_percentage, 2)
    return distracted_percentage


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
        #model_path = pathlib.Path('model.h5')
        #このモデルは顔が映りこむとnot_distracted判定になるため使えない
        #model_path = pathlib.Path('2_11_model.h5')
        #model_path = pathlib.Path('2_12_forth_model.h5')
        #たぶんこれが最終版
        model_path = pathlib.Path('2_12_fifth_model.h5')

        self.model = tf.keras.models.load_model(model_path, compile=False)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        self.frame_count += 1

        if (self.frame_count % 5) != 0:
            return {}

        img = inputs["img"]
        bboxes = inputs["bboxes"]

        if bboxes is not None:
            for bbox in bboxes:
                x_max, x_min, y_max, y_min = map_bbox_to_image_coords(img, bbox)
                x_min, y_min = int(x_min), int(y_min)

                sliced_img = slice_img(img, bbox)
                resized_img_tensor = resize_img(sliced_img)
                img_percentage = predict_img_percentage(resized_img_tensor, self.model)
                str_img_percentage = str(img_percentage)

                if img_percentage >= 80:
                    cv2.putText(
                        img=img,
                        text=str_img_percentage,
                        org=(x_min, y_min),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(0, 0, 255),
                        thickness=3,
                    )

            return {}
