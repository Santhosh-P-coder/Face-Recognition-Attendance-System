import cv2
import pickle
import time
from RPLCD.i2c import CharLCD

# ---------- LCD SETUP ----------
lcd = CharLCD('PCF8574', 0x27)

lcd.clear()
lcd.write_string("System Starting")
time.sleep(2)

# ---------- LOAD MODEL ----------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

with open("trainer/labels.pickle", "rb") as f:
    labels = pickle.load(f)

# ---------- FACE DETECTOR ----------
faceCascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

font = cv2.FONT_HERSHEY_SIMPLEX

# ---------- CAMERA ----------
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    lcd.clear()
    lcd.write_string("Camera Error")
    print("Camera not opened")
    exit()

print("Face Recognition Started")

lcd.clear()
lcd.write_string("Face System")
lcd.cursor_pos = (1, 0)
lcd.write_string("Scanning...")

unknown_start_time = None

while True:
    ret, img = cam.read()

    if not ret:
        lcd.clear()
        lcd.write_string("Camera Error")
        print("Camera error")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        unknown_start_time = None

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        accuracy = round(100 - confidence)

        # ---------- RECOGNIZED PERSON ----------
        if accuracy >= 50:
            name = labels.get(id, "Unknown")

            print(f"Recognized: {name} ({accuracy}%)")

            lcd.clear()
            lcd.write_string("Welcome")
            lcd.cursor_pos = (1, 0)
            lcd.write_string(name[:16])

            cv2.putText(img, f"{name} {accuracy}%",
                        (x + 5, y - 10),
                        font, 1, (0, 255, 0), 2)

            cv2.imshow("Face Recognition", img)
            cv2.waitKey(3000)

            cam.release()
            cv2.destroyAllWindows()
            lcd.clear()
            exit()

        # ---------- WRONG PERSON ----------
        else:
            if unknown_start_time is None:
                unknown_start_time = time.time()

            unknown_duration = time.time() - unknown_start_time

            lcd.clear()
            lcd.write_string("Unknown Person")
            lcd.cursor_pos = (1, 0)
            lcd.write_string(f"Time:{int(unknown_duration)}s")

            cv2.putText(img, "Unknown",
                        (x + 5, y - 10),
                        font, 1, (0, 0, 255), 2)

            cv2.putText(img, f"Time: {int(unknown_duration)} sec",
                        (x + 5, y + h + 30),
                        font, 0.8, (0, 0, 255), 2)

            if unknown_duration > 5:
                print("Wrong Person Detected")

                lcd.clear()
                lcd.write_string("Wrong Person")
                lcd.cursor_pos = (1, 0)
                lcd.write_string("Detected")

                cv2.imshow("Face Recognition", img)
                cv2.waitKey(3000)

                cam.release()
                cv2.destroyAllWindows()
                lcd.clear()
                exit()

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
lcd.clear()