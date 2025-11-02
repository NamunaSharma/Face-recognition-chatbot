# from fastapi import FastAPI
import cv2
import os
import numpy as np
import face_recognition


font = cv2.FONT_HERSHEY_DUPLEX

# app = FastAPI()
# @app.get("/")
# def read_rot():
#     return {"message": "Hello, World!"}


# @app.post('/register')
# def register_user():  


# @app.post('/login')
# def login_user():  

known_face_encodings =[]
known_face_names = []

image1 = face_recognition.load_image_file("images/face1.jpg")
face_encoding1 = face_recognition.face_encodings(image1)[0]
known_face_encodings.append(face_encoding1)
known_face_names.append("Person One")

image2 = face_recognition.load_image_file("images/face2.jpg")
face_encoding2 = face_recognition.face_encodings(image2)[0]
known_face_encodings.append(face_encoding2)
known_face_names.append("Person Two")


face_locations =[]
face_encodings = []
face_names = []

# Video capture from webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    rgb_small_frame = cv2.cvtColor(rgb_small_frame , cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Display the results
    for (top,right,bottom,left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Show the processed frame from within the while loop
    cv2.imshow('Video', frame)
    # Use correct function name 'waitKey' and keep this check inside the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Exit the while loop when 'q' is pressed
        break

# Release resources after the loop ends
video_capture.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()