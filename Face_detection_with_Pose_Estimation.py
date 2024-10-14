import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8)

cap = cv2.VideoCapture('input_video.mp4')  # 0 for the default camera
fourcc = cv2.VideoWriter_fourcc(*'mprv')  # Codec for MP4 writing
output_filename = 'output_video.mp4'  # Output video file name

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Process the image for face detection
    face_results = face_detection.process(image_rgb)

    # Process the image for pose estimation
    pose_results = pose.process(image_rgb)

    # Draw face detection results
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            cv2.rectangle(frame, 
                          (int(bboxC.xmin * w), int(bboxC.ymin * h)),
                          (int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)),
                          (0, 255, 0), 2)

    # Draw pose landmarks
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    out.write(frame)

    # Display the image
    cv2.imshow('MediaPipe Combined Models', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
