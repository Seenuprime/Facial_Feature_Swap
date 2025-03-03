import mediapipe as mp
import cv2 as cv
import numpy as np
import warnings

warnings.filterwarnings('ignore')

mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

image = cv.imread('face.jpg')

img_height, img_width = image.shape[:2]
image_result = face.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

image_landamrks = image_result.multi_face_landmarks[0].landmark
left_eye_indices = list(range(37, 44))
left_eye_image_pxl = np.array([(int(image_landamrks[i].x * img_width), int(image_landamrks[i].y * img_height)) for i in left_eye_indices], dtype=np.int32)

print(left_eye_image_pxl)
# cap = cv.VideoCapture('video.mp4')

# while cap.isOpened():
#     ret, o_frame = cap.read()

#     if not ret:
#         break

#     o_frame = cv.flip(o_frame, 1)
#     h, w = o_frame.shape[0] // 2, o_frame.shape[1] // 2
#     frame = cv.resize(o_frame, (w, h))

#     results = face.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

#     if results:
#         for face_landmarks in results.multi_face_landmarks:
#             for i, landmarks in enumerate(face_landmarks.landmark):
#                 land_width = int(landmarks.x * w)
#                 land_height = int(landmarks.y * h)

#                 # print(land_width, land_height)
#                 cv.circle(frame, (land_width, land_height), 2, (255, 0, 0), -1)
                
    
#     cv.imshow("Image", frame)
#     if cv.waitKey(0) & 0xFF == ord('q'):
#         break
    

# cap.release()
# cv.destroyAllWindows()
