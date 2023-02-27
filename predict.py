import tensorflow as tf


class Predict:
    def __init__(self):
        model_path = 'lite_model_2.tflite'
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()
        self.input_index = self.model.get_input_details()[0]['index']
        self.output_index = self.model.get_output_details()[0]['index']

    def predict_percentage(self, frame_tensors):
        self.model.set_tensor(self.input_index, frame_tensors)
        self.model.invoke()
        prediction = self.model.get_tensor(self.output_index)
        prediction_value = prediction[0]
        #prediction_value = prediction_value.numpy()
        distracted_percentage = 100 * (1 - prediction_value)

        return distracted_percentage
