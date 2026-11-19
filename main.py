import cv2 as cv
import face_recognition
import mediapipe as mp
import time
import os
import threading
import math
cap = cv.VideoCapture(0)
class Live_face_compare:
    def __init__(self):
        self.recognized = False
        self.modified_known_faces = []
        self.face_locations_to_draw = []
        self.known_processed_faces = []
        self.has_blinked=False
        self.current_frame=None
        self.lock = threading.Lock()
        self.running=True

        lisdir = os.listdir("images")
        known_faces = []
        for i in lisdir:
            for j in range(len(i) - 1):

                if i[j] == ".":

                    if i[j + 1:j + 5] == "png" or i[j + 1:j + 5] == "jpg":
                        known_faces.append("./images/{}".format(i[0:j+4]))

        for i in known_faces:
            for j in range(10,len(i),1):
                if i[j]==".":
                    self.modified_known_faces.append(i[9:j])
        #print(self.known_faces)
        for face in known_faces:
            image_path = face
            image = cv.imread(image_path)

            if image is None:
                print("error no image has been read")
            else:
                rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_image)
                if len(encodings) > 0:
                    self.known_processed_faces.extend(encodings)
        #print(self.known_processed_faces)
    def compare_background(self):
        count=0
        while self.running:
            #self.recognized=False
            with self.lock:
                if self.current_frame is None:
                    continue
                frame_to_process = self.current_frame.copy()
            rgb_image1 = cv.cvtColor(frame_to_process, cv.COLOR_BGR2RGB)
            face_locations=face_recognition.face_locations(rgb_image1)
            print("#######################")
            print(face_locations)
            print("#######################")

            #print(f"num of faces in frame: {len(face_locations)}")
            face_encodings = face_recognition.face_encodings(rgb_image1,face_locations)
            current_recognized_faces = []
            for loc, encoding in zip(face_locations, face_encodings):

                if len(self.known_processed_faces) > 0:
                    matches = face_recognition.compare_faces(self.known_processed_faces, encoding)
                    #print(f" (Matches): {matches}")
                    for idx, match in enumerate(matches):
                        if match:
                            current_recognized_faces.append((loc, self.modified_known_faces[idx]))
                            count+=1
                            break



            if count < 4:
                continue
            else:
                with self.lock:
                    self.face_locations_to_draw = current_recognized_faces
                    self.recognized = len(current_recognized_faces) > 0
                    print("count: ", count)
                    print("recognition status: ", self.recognized)
                    break
        if self.running:
            self.eye_dis()
                            # print(self.recognized)
                #else:
                    #print("list of faces is emptyً!")

    def eye_dis(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        LEFT_EYE = {
            "u1": 159,
            "l1": 145,
            "u2": 158,
            "l2": 153,
            "left": 33,
            "right": 133,
        }
        RIGHT_EYE = {
            "u1": 386,
            "l1": 374,
            "u2": 385,
            "l2": 380,
            "left": 362,
            "right": 263,
        }

        eye_closed = False
        start_time = 0
        blink_duration = 0.0
        EAR_THRESHOLD = 0.17
        closed_frames_counter = 0
        MIN_CLOSED_FRAMES = 2

        def calculate_ear(landmarks,eye_points, w, h):
            p = {}
            for key, idx in eye_points.items():
                pt = landmarks.landmark[idx]
                p[key] = (int(pt.x * w), int(pt.y * h))


            v1 = math.dist(p["u1"], p["l1"])
            v2 = math.dist(p["u2"], p["l2"])

            h_dist = math.dist(p["left"], p["right"])

            ear = (v1 + v2) / (2.0 * h_dist)
            return ear, p
        while self.running:
            with self.lock:
                if self.current_frame is None:
                    continue
                frame_to_processm = self.current_frame.copy()
            frame = cv.flip(frame_to_processm, 1)
            h, w, _ = frame.shape
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_frame)
            current_time = time.time()

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_ear, lp = calculate_ear(face_landmarks,LEFT_EYE, w, h)
                    right_ear, rp = calculate_ear(face_landmarks, RIGHT_EYE, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0

                    '''
                    upper_point = face_landmarks.landmark[UPPER_EYELID_IDX]
                    lower_point = face_landmarks.landmark[LOWER_EYELID_IDX]

                    u_x, u_y = int(upper_point.x * w), int(upper_point.y * h)
                    l_x, l_y = int(lower_point.x * w), int(lower_point.y * h)
                    distance = np.sqrt((u_x - l_x) ** 2 + (u_y - l_y) ** 2)
                    '''
                    if avg_ear < EAR_THRESHOLD:
                        closed_frames_counter+=1
                        if closed_frames_counter == MIN_CLOSED_FRAMES and not eye_closed:
                            eye_closed = True
                            start_time = current_time
                    else:
                        if eye_closed:
                            eye_closed = False
                            end_time = current_time
                            blink_duration = end_time - start_time
                            print("blink_duration: ",blink_duration)

                            if 0.05<blink_duration<0.1:
                                with self.lock:
                                    self.has_blinked=True
                                print("user has blinked")
                        closed_frames_counter=0


    def video_show(self):
        while cap.isOpened():
            rel,frame=cap.read()
            if not rel:
                break
            with self.lock:
                self.current_frame = frame
                is_recognized=self.recognized
                faces_to_draw = list(self.face_locations_to_draw)
                blinked=self.has_blinked
            frame = cv.flip(frame, 1)
            if is_recognized:
                for loc, name in faces_to_draw:
                    top, right, bottom, left = loc
                    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv.putText(frame, name, (left, top - 10),
                               cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 1)
                    if not blinked:
                        cv.putText(frame, "blink", (left, bottom+20),
                               cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0),3)

                    else:
                        cv.putText(frame, "live face ID has been executed successfully", (left, bottom+20),
                                   cv.FONT_HERSHEY_COMPLEX,0.4, (255, 0, 0), 3)

            cv.imshow("face", frame)
            if cv.waitKey(1) == ord('q'):
                with self.lock:
                    self.running=False
                break
        cap.release()
        cv.destroyAllWindows()








lfc=Live_face_compare()
threed1=threading.Thread(target=lfc.compare_background)

threed2=threading.Thread(target=lfc.video_show)
threed1.start()
threed2.start()
threed1.join()
threed2.join()