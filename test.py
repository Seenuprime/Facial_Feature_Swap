import mediapipe as mp
import cv2 as cv
import numpy as np
import warnings

warnings.filterwarnings('ignore')

mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


image = cv.imread('face.jpg')
if image is None:
    print("Error: Image not found. Check the path.")
    exit()

img_height, img_width = image.shape[:2]
image_result = face.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

image_landmarks = image_result.multi_face_landmarks[0].landmark
left_eye_indices = [33, 160, 158, 133, 153, 144]  # Left eye landmarks from MediaPipe
left_eye_image_pxl = np.array([(int(image_landmarks[i].x * img_width), int(image_landmarks[i].y * img_height)) 
                               for i in left_eye_indices], dtype=np.int32)

# Function to create a mask for the feature
def get_feature_mask(img, points, padding=20):
    hull = cv.convexHull(points)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.fillConvexPoly(mask, hull, 255)
    mask = cv.dilate(mask, np.ones((padding, padding), np.uint8))
    return mask

mask_img = get_feature_mask(image, left_eye_image_pxl)
left_eye_region_image = cv.bitwise_and(image, image, mask=mask_img)

cap = cv.VideoCapture('video.mp4')  # Replace with your video path

while cap.isOpened():
    ret, o_frame = cap.read()
    if not ret:
        break

    o_frame = cv.flip(o_frame, 1)
    h, w = o_frame.shape[0] // 2, o_frame.shape[1] // 2
    frame = cv.resize(o_frame, (w, h))

    results = face.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        # for face_landmarks in results.multi_face_landmarks:
        #     # for i, landmarks in enumerate(face_landmarks.landmark):
        #     #     land_width = int(landmarks.x * w)
        #     #     land_height = int(landmarks.y * h)

        #     #     # print(land_width, land_height)
        #     #     cv.circle(frame, (land_width, land_height), 2, (255, 0, 0), -1)

        frame_landmark = results.multi_face_landmarks[0].landmark
        left_eye_frame_pxl = np.array([(int(frame_landmark[i].x * w), int(frame_landmark[i].y * h)) 
                                       for i in left_eye_indices], dtype=np.int32)

        # Warp image eye to video face
        src_pts = left_eye_image_pxl[[0, 2, 4]].astype(np.float32)  # Key points from image
        dst_pts = left_eye_frame_pxl[[0, 2, 4]].astype(np.float32)  # Key points from video
        M = cv.getAffineTransform(src_pts, dst_pts)

        warped_eye = cv.warpAffine(left_eye_region_image, M, (w, h))
        warped_mask = cv.warpAffine(mask_img, M, (w, h))

        # Blend onto video frame
        center = np.mean(left_eye_frame_pxl, axis=0).astype(int)
        try:
            frame = cv.seamlessClone(warped_eye, frame, warped_mask, 
                                    (center[0], center[1]), cv.NORMAL_CLONE)
        except cv.error as e:
            print(f"Blending error: {e}. Using original frame.")
            # Skip blending if it fails (e.g., mask misalignment)

        # # Optional: Draw video landmarks for debugging
        # for i in left_eye_indices:
        #     cv.circle(frame, (int(frame_landmark[i].x * w), int(frame_landmark[i].y * h)), 
        #              2, (255, 0, 0), -1)

    cv.imshow("Image", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):  # 1ms delay for smooth playback
        break

cap.release()
cv.destroyAllWindows()