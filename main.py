import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image
import dlib
import sys

emotion_labels = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}


# * load model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# * load weights
model.load_weights("model/model_weights.h5")

# * dlib face detector
face_decector = dlib.get_frontal_face_detector()

# * video/webcam
# video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture('./examples/acting.mp4', cv2.CAP_ANY)

COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)

if (video_capture.isOpened() == False):
    print("No video or camera detected")
    sys.exit()

while True:
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (1280, 720))

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_decector(frame, 0)

    for face_rect in faces:
        x1 = face_rect.left()
        y1 = face_rect.top()
        x2 = face_rect.right()
        y2 = face_rect.bottom()

        roi = gray_frame[y1:y2, x1:x2]

        # convert image to tensor shape
        im = Image.fromarray(roi)
        im = im.resize((48, 48))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)

        # predict
        prediction = model.predict(img_array)
        class_idx = prediction.argmax(axis=-1)[0]
        label = emotion_labels[class_idx]

        # draw rectangle and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, COLOR, 2)

    cv2.imshow('Detection (Quit = ESC)', frame)

    # * exit
    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC
        break

video_capture.release()
cv2.destroyAllWindows()
