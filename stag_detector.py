import pyStag as stag
import cv2
import numpy as np
import pandas as pd
import math

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

def find_stag(frame):
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det = stag.Detector(11, 7 , False)
    det.detect(grayimg)
    coners = det.getContours()
    ids = det.getIds()
    unique_ids = []
    unique_coners = []

    # remove duplicate ids and coners
    if len(ids) != 0:
        unique_dict = dict(zip(ids, coners))
        for id, coner in unique_dict.items():
            unique_ids.append(id)
            unique_coners.append(coner)
        # print(f'unique_dict: {unique_dict}\n')
        # print(f'unique_ids:{unique_ids} \t unique_coners:{unique_coners}\n')

    centers = []
    i = 0
    O_center = []
    tvecs = []
    rvecs = []
    for bbx, id in zip(unique_coners, unique_ids):
        top_left, top_right, bottom_right, bottom_left = bbx
        
        top_left = [int (x) for x in top_left]
        top_right = [int (x) for x in top_right]
        bottom_right = [int (x) for x in bottom_right]
        bottom_left = [int (x) for x in bottom_left]
        center = [int((top_left[0] + bottom_right[0]) / 2.0), int((top_left[1] + bottom_right[1]) / 2.0)]

        bbx_for_axis = np.rint(np.array(bbx).reshape(1,4,2)) 
        rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(bbx_for_axis, 0.08, intrinsic_matrix, distortion_matrix)
        # print(f'rvec:{rvec}\t rvec[0]:{rvec[0]}')
        tvec_added_id = np.append(tvec,id).reshape(1,1,4)
        # print(f'{tvec_added_id.shape}\t {tvec_added_id}')
        tvecs.append(tvec_added_id)
        rvecs.append(rvec)

        cv2.drawFrameAxes(frame, intrinsic_matrix, distortion_matrix, rvec, tvec, 0.04)

        frame = cv2.line(frame,top_left, top_right,(255,255,0),1)
        frame = cv2.line(frame,top_right,bottom_right,(255,225,0),1)
        frame = cv2.line(frame,bottom_right,bottom_left,(255,0,225),1)
        frame = cv2.line(frame,bottom_left, top_left,(255,0,225),1)
        frame = cv2.line(frame, center, center, (255,0,0),3)
        frame = cv2.putText(frame, "{}".format(id),top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,225), 2)

        # if i % 4 == 3:
        if id == 0:
            O_center = center
            O_center.append(id)
        else :
            center.append(id)
            centers.append(center)

        i += 1

    return frame, O_center ,centers, tvecs, rvecs

def cal_distance(frame, origin_center ,centers, tvecs, rvecs):
    t = np.array(tvecs)
    r = np.array(rvecs)
    # if len(t) != 0:
    #     print(f't:{t}\t len(t):{len(t)}\t t[0]:{t[0]}\t t[0][0]:{t[0][0][0][3]}\n')
    
    for i in range(0, len(t)):
        if(int(t[i][0][0][3]) == 0):
            tvec0 = t[i][:,:,:3]
            # print(f't[i]:{t[i]}\t tvec0: {tvec0} \n')
            R_ct = np.matrix(cv2.Rodrigues(r[i])[0])
            R_ct = R_ct.T
            roll, pitch, yaw = rotationMatrixToEulerAngles(R_flip*R_ct)
            A = ([math.cos(roll),math.cos(pitch),math.cos(yaw)])
            print(f'tvec0: {tvec0}\n')
            print(f'A: {A}\n')

        if(int(t[i][0][0][3]) == 1):
            tvec1 = t[i][:,:,:3]
            # print(f't[i]:{t[i]}\t tvec0: {tvec0} \n')
            R_ct = np.matrix(cv2.Rodrigues(r[i])[0])
            R_ct = R_ct.T
            roll, pitch, yaw = rotationMatrixToEulerAngles(R_flip*R_ct)
            B = ([math.cos(roll),math.cos(pitch),math.cos(yaw)])
            print(f'tvec1: {tvec1}\n')
            print(f'B: {B}\n')

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

    for center in centers:
        if(len(center) !=0 and len(origin_center) != 0):
            frame = cv2.line(frame, (origin_center[0],origin_center[1]), (center[0], center[1]), (255,255,0), 2)

    return frame


cam = cv2.VideoCapture(0)
intrinsic_matrix = np.loadtxt("intrinsic_matrix.txt", dtype=float)
distortion_matrix = np.loadtxt("distortion_matrix.txt", dtype=float)


while(True):
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame, origin_center ,stag_centers, tvecs, rvecs = find_stag(frame)
    frame = cal_distance(frame, origin_center ,stag_centers, tvecs, rvecs) 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        

    cv2.imshow("test", frame)

    if cv2.waitKey(1) == ord('q'):
        break

