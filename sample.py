'''
Sample Command:-
python detect_aruco_video.py --type DICT_5X5_100 --camera True
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video test_video.mp4 -a 25 -k ./calibration_matrix.npy -d ./distortion_coefficients.npy
'''

from turtle import delay
import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys
import math
import time

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--camera", help="Set to True if using webcam")
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
ap.add_argument("-a", "--aruco_dim", required=True, help="ArUco tag dimension")
args = vars(ap.parse_args())


if args["camera"].lower() == "true":
    video = cv2.VideoCapture(0)
    time.sleep(2.0)
    
else:
    if args["video"] is None:
        print("[Error] Video file location is not provided")
        sys.exit(1)

    video = cv2.VideoCapture(args["video"])

if ARUCO_DICT.get(args["type"], None) is None:
    print(f"ArUCo tag type '{args['type']}' is not supported")
    sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
calibration_matrix_path = args["K_Matrix"]
distortion_coefficients_path = args["D_Coeff"]
k = np.load(calibration_matrix_path)
d = np.load(distortion_coefficients_path)
arucoParams = cv2.aruco.DetectorParameters_create()

while True:
    ret, frame = video.read()
    time.sleep(0.05)
    
    if ret is False:
        break


    h, w, _ = frame.shape

    width=1000
    height = int(width*(h/w))
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    detected_markers = aruco_display(corners, ids, rejected, frame)
    if len(corners) > 0:
        #print(corners) #posizione degli angoli del marker
        #print(len(ids))
        if (len(ids) == 2):
            for i in range(0, len(ids)):
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], float(args["aruco_dim"]), k,d)
                if (i == 0):
                    print("Vettore 1")
                    tvec0 = tvec
                    R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                    R_ct = R_ct.T
                    roll, pitch, yaw = rotationMatrixToEulerAngles(R_flip*R_ct)
                    A = ([math.cos(roll),math.cos(pitch),math.cos(yaw)])
                    print(tvec0)
                    print(A)
                if (i==1):
                    print("Vettore 2")
                    tvec1 = tvec
                    #B = ([rvec[0][0][0],rvec[0][0][1],rvec[0][0][2]])
                    R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                    R_ct = R_ct.T
                    roll, pitch, yaw = rotationMatrixToEulerAngles(R_flip*R_ct)
                    B = ([math.cos(roll),math.cos(pitch),math.cos(yaw)])
                    print(tvec1)
                    print(B)

                    # primo metodo per il calcolo della distanza
                    tvec0_x = tvec0[0][0][0]
                    tvec0_y = tvec0[0][0][1]
                    tvec0_z = tvec0[0][0][2]
                    tvec1_x = tvec1[0][0][0]
                    tvec1_y = tvec1[0][0][1]
                    tvec1_z = tvec1[0][0][2]
                    dist1 = math.sqrt(pow((tvec0_x-tvec1_x),2)+pow((tvec0_y-tvec1_y),2)+pow((tvec0_z-tvec1_z),2))
                    distanza1= "Dist=%4.0f"%(dist1)
                    #secondo metodo per il calcolo della distanza
                    #
                    distanza= "Dist=%4.0f"%(np.linalg.norm(tvec1-tvec0))
                    cv2.putText(frame, distanza1,(50, 100),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2, cv2.LINE_4)
                    dot_product = np.dot(A,B,out=None)
                    normA = (np.linalg.norm(A))
                    normB = (np.linalg.norm(B))
                    cos_angolo = dot_product/(normA*normB)
                    angolo_rad = np.arccos(cos_angolo)
                    angolo_deg = np.rad2deg(angolo_rad)
                    if (angolo_deg > 90):
                        angolo_deg = 180 - angolo_deg
                    ang = "Ang=%4.1f"%(angolo_deg)
                    cv2.putText(frame,ang ,(50, 150),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2, cv2.LINE_4)


    cv2.imshow("Image", detected_markers)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video.release()