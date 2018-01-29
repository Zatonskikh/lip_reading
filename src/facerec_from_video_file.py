import face_recognition
import cv2

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file

def get_lips(top_lip, bottom_lip):

    return top_lip[0:7] + bottom_lip[0:5] + top_lip[7:][::-1] + bottom_lip[8:11][::-1]

def face_recognition_custom(path, word=None):
    input_movie = cv2.VideoCapture("../data/" + path)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    heights = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_movie.get(cv2.CAP_PROP_FPS))

    # Create an output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('output.avi', fourcc, fps, (width, heights))

    # Load some sample pictures and learn how to recognize them.
    lmm_image = face_recognition.load_image_file("efim.jpg")
    lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]


    known_faces = [
        lmm_face_encoding,
    ]

    # Initialize some variables
    frame_number = 0
    i = 0
    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1
        if not ret:
            break

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            #match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
            match = face_recognition.compare_faces([lmm_face_encoding], face_encoding)

            # If you had more than 2 faces, you could make this logic a lot prettier
            # but I kept it simple for the demo
            name = None
            if match[0]:
                i+=1
                name = "efim"
                print ("found face {}".format(i))
                face_landmarks = face_recognition.face_landmarks(frame, face_locations)
                lips = get_lips(face_landmarks[0]["top_lip"], face_landmarks[0]["bottom_lip"])
                for x, y in lips:
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            else:
                raise Exception("NOT VALID PEOPLE")

            face_names.append(name)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 5), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom + 6), font, 3.5, (255, 255, 255), 3)
            cv2.putText(frame, word, (30, bottom + 150), font, 3.5, (255, 255, 255), 3)

        # Write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, length))
        output_movie.write(frame)

    # All done!
    input_movie.release()
