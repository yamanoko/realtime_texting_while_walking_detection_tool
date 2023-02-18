import tensorflow as tf


def slice_image(image, x, y, w, h):
    sliced_image = image[int(y):int(y + h), int(x):int(x + w)]

    return sliced_image


def resize_image(image):
    image_tensor = tf.constant(image)
    resized_image_tensor = tf.image.resize_with_pad(image_tensor, 256, 256)

    return resized_image_tensor


def change_image_shape(frame, bounding_boxes):
    changed_images = []
    for (x, y, w, h) in bounding_boxes:
        sliced_image = slice_image(frame, x, y, w, h)
        resized_image_array = resize_image(sliced_image)
        changed_images.append(resized_image_array)

    changed_image_tensor = tf.stack(changed_images)
    print(changed_image_tensor.shape)

    return changed_image_tensor
