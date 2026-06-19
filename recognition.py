import cv2
import pickle
import time
from picamera2 import Picamera2
from RPLCD.i2c import CharLCD

lcd = CharLCD("PCF8574", 0x27)
lcd.clear()
lcd.write_string("System Starting")
time.sleep(2)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

with open("trainer/labels.pickle", "rb") as f:
    labels = pickle.load(f)

faceCascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)
picam2.start()
time.sleep(2)

lcd.clear()
lcd.write_string("Scanning...")

unknown_start_time = None

while True:
    img = picam2.capture_array()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        unknown_start_time = None

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        accuracy = round(100 - confidence)

        if accuracy >= 50:
            name = labels.get(id, "Unknown")

            lcd.clear()
            lcd.write_string("Welcome")
            lcd.cursor_pos = (1, 0)
            lcd.write_string(name[:16])

            time.sleep(10)
            picam2.stop()
            lcd.clear()
            exit()

        else:
            if unknown_start_time is None:
                unknown_start_time = time.time()

            unknown_duration = time.time() - unknown_start_time

            lcd.clear()
            lcd.write_string("Unknown")
            lcd.cursor_pos = (1, 0)
            lcd.write_string(str(int(unknown_duration)) + " sec")

            if unknown_duration > 5:
                lcd.clear()
                lcd.write_string("Wrong Person")
                lcd.cursor_pos = (1, 0)
                lcd.write_string("Detected")

                time.sleep(10)
                picam2.stop()
                lcd.clear()
                exit()

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) == 27:
        break

picam2.stop()
cv2.destroyAllWindows()
lcd.clear()