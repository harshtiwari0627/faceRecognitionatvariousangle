import face_recognition
import cv2
import numpy as np


b = face_recognition.load_image_file('Images/h.jpg')  # Edit the image location 
c = face_recognition.face_encodings(b)[0]

known_face_encoding=[]
known_face_encoding.append(c)
      
know_face_names = []
know_face_names.append("Suspect Found !!!")

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for(top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_face_encoding, face_encodings)

        name = "Unknown"

        if True in matches :
            first_match_index = matches.index(True)
            name = know_face_names[first_match_index]

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 80, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('video', frame)

    #   0 to quit !
    #   if (cv2.Waitkey(1) & 0xFF == ord('q') ):
    #   break
    cv2.waitKey(0)
video_capture.release()
cv2.destroyAllWindows()
