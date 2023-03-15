import cv2

video1 = "230223_aruco_result.mp4"
video2 = "aruco_tracking.mp4"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

cap1 = cv2.VideoCapture(video1)
cap2 = cv2.VideoCapture(video2)

result = cv2.VideoWriter('compare_tracking.mp4',cv2.VideoWriter_fourcc(*'MP4V'),10, (CAMERA_WIDTH*2, CAMERA_HEIGHT))

while True:
    rat1, frame1 = cap1.read()
    rat2, frame2 = cap2.read()

    add = cv2.hconcat([frame1, frame2])

    cv2.imshow('test', add)

    result.write(add)

    if cv2.waitKey(1) == ord('q'):
        break