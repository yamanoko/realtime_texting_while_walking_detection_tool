import cv2
from detect_human import DetectHuman
import change_image_shape
from predict import PredictDistractedPeople


PredictDistractedPeople = PredictDistractedPeople()

def main():
    cap = cv2.VideoCapture(0)
    frame_width = cap.get(3)
    frame_height = cap.get(4)
    cap.set(cv2.CAP_PROP_FPS, 1)
    human_detect = DetectHuman(frame_width=frame_width, frame_height=frame_height)

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        bounding_boxes = human_detect.detect(frame)

        if bounding_boxes is None:
            cv2.imshow('camera', frame)
            continue

        changed_image_tensor = change_image_shape.change_image_shape(frame, bounding_boxes)
        distracted_percentages = PredictDistractedPeople.predict(changed_image_tensor)
        distracted_mask = (distracted_percentages >= 80)
        distracted_boxes = bounding_boxes[distracted_mask]

        for (x, y, w, h) in distracted_boxes:
            cv2.rectangle(img=frame,
                          plt1=(x, y),
                          plt2=(x + w, y + h),
                          color=(0, 255, 0),
                          thickness=3
                          )

        cv2.imshow('camera', frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC Key to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
