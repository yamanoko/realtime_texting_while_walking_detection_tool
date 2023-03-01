import tensorflow as tf


def slice_image(image, bounding_box):
    y_min = int(bounding_box[0])
    x_min = int(bounding_box[1])
    y_max = int(bounding_box[2])
    x_max = int(bounding_box[3])

    sliced_image = image[y_min:y_max, x_min:x_max]

    return sliced_image


def resize_image(image):
    image_tensor = tf.constant(image)
    resized_image_tensor = tf.image.resize_with_pad(image_tensor, 256, 256)

    return resized_image_tensor


def change_image_shape(frame, bounding_boxes):
    changed_images = []
    for bounding_box in bounding_boxes:
        sliced_image = slice_image(frame, bounding_box)
        resized_image_array = resize_image(sliced_image)
        changed_images.append(resized_image_array)

    changed_image_tensor = tf.stack(changed_images)

    return changed_image_tensor
