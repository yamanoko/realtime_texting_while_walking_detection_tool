import tensorflow as tf


class Predict:
    def __init__(self, model_path):
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()
        self.input_index = self.model.get_input_details()[0]['index']
        self.output_index = self.model.get_output_details()[0]['index']

    def predict(self, frame):
        self.model.set_tensor(self.input_index, frame)
        self.model.invoke()
        prediction = self.model.get_tensor(self.output_index)
        return prediction


class PredictDistractedPeople(Predict):
    def __init__(self, model_path='lite_model_2.tflite'):
        super().__init__(model_path)

    def predict(self, frame):
        prediction = super().predict(frame)
        prediction_value = prediction[0]
        distracted_percentage = 100 * (1 - prediction_value)

        return distracted_percentage
