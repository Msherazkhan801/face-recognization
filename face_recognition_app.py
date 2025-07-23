import numpy as np
import face_recognition
import os
import cv2
from datetime import datetime

# Path to folder with known images
path = 'Images'
images = []
names = []

# Load images with checks
for file in os.listdir(path):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"⛔ Skipping non-image file: {file}")
        continue
    img = cv2.imread(f'{path}/{file}')
    if img is None:
        print(f"❌ Could not load image: {file}")
        continue
    images.append(img)
    names.append(os.path.splitext(file)[0])

# Encode known faces
def findEncodings(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodings.append(encode)
        except:
            print("⚠️ No face found in one image. Skipping.")
    return encodings

# Mark attendance in CSV
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        recorded = [line.split(',')[0] for line in f.readlines()]
        if name not in recorded:
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')
            date_str = now.strftime('%d/%m/%Y')
            f.write(f'\n{name},{time_str},{date_str}')

# Start
known_encodings = findEncodings(images)
print('✅ Encoding complete.')

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to read from webcam.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, faces)

    for face_encoding, face_location in zip(encodings, faces):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        match_index = np.argmin(face_distances)

        if matches[match_index]:
            name = names[match_index].upper()
            y1, x2, y2, x1 = [v * 4 for v in face_location]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)

    cv2.imshow('Webcam Face Recognition', frame)

    if cv2.waitKey(10) == 13:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()
