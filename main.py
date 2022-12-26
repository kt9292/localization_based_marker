import pyStag as stag
import cv2
import numpy as np

def find_aruco(frame, marker_size=4, total_markers=250, draw=True):
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    aruco_dict = cv2.aruco.Dictionary_get(key)
    aruco_param = cv2.aruco.DetectorParameters_create()
    bbox, ids, _ = cv2.aruco.detectMarkers(grayimg, aruco_dict, parameters=aruco_param)

    centers = []
    
    if len(bbox) != 0:
        for bbx, id in zip(bbox[0],ids):
            top_left, top_right, bottom_right, bottom_left = bbx
            
            top_left = [int (x) for x in top_left]
            top_right = [int (x) for x in top_right]
            bottom_right = [int (x) for x in bottom_right]
            bottom_left = [int (x) for x in bottom_left]
            center = [int((top_left[0] + bottom_right[0]) / 2.0), int((top_left[1] + bottom_right[1]) / 2.0)]
            centers.append(center)            

            print(type(bbox[0]))
            rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(bbox[0], 0.1, intrinsic_matrix, distortion_matrix)
            cv2.drawFrameAxes(frame, intrinsic_matrix, distortion_matrix, rvec, tvec, 0.05)

            frame = cv2.line(frame,top_left, top_right,(255,255,0),1)
            frame = cv2.line(frame,top_right,bottom_right,(255,225,0),1)
            frame = cv2.line(frame,bottom_right,bottom_left,(255,0,225),1)
            frame = cv2.line(frame,bottom_left, top_left,(255,0,225),1)
            frame = cv2.line(frame, center, center, (255,0,0),3)
            frame = cv2.putText(frame, "{}".format(id[0]),top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,225), 2)

    
    return frame, centers



def find_stag(frame):
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det = stag.Detector(11, 7 , False)
    det.detect(grayimg)
    coners = det.getContours()
    ids = det.getIds()

    centers = []
    i = 0
    O_center = []
    for bbx, id in zip(coners, ids):
        top_left, top_right, bottom_right, bottom_left = bbx
        
        top_left = [int (x) for x in top_left]
        top_right = [int (x) for x in top_right]
        bottom_right = [int (x) for x in bottom_right]
        bottom_left = [int (x) for x in bottom_left]
        center = [int((top_left[0] + bottom_right[0]) / 2.0), int((top_left[1] + bottom_right[1]) / 2.0)]

        bbx_for_axis = np.rint(np.array(bbx).reshape(1,4,2)) 

        rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(bbx_for_axis, 0.08, intrinsic_matrix, distortion_matrix)
        cv2.drawFrameAxes(frame, intrinsic_matrix, distortion_matrix, rvec, tvec, 0.04)

        frame = cv2.line(frame,top_left, top_right,(255,255,0),1)
        frame = cv2.line(frame,top_right,bottom_right,(255,225,0),1)
        frame = cv2.line(frame,bottom_right,bottom_left,(255,0,225),1)
        frame = cv2.line(frame,bottom_left, top_left,(255,0,225),1)
        frame = cv2.line(frame, center, center, (255,0,0),3)
        frame = cv2.putText(frame, "{}".format(id),top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,225), 2)

        if i % 4 == 3:
            if id == 0:
                O_center = center
                O_center.append(id)
            else :
                center.append(id)
                centers.append(center)

        i += 1

    return frame, O_center ,centers

def cal_distance(frame, origin_center ,centers):
    camera_dis = 0
    real_dis = 0
    for center in centers:
        # camera_dis = 
        if(len(center) !=0 and len(origin_center) != 0):
            frame = cv2.line(frame, (origin_center[0],origin_center[1]), (center[0], center[1]), (255,0,0), 5)

    return frame


cam = cv2.VideoCapture(0)
intrinsic_matrix = np.loadtxt("intrinsic_matrix.txt", dtype=float)
distortion_matrix = np.loadtxt("distortion_matrix.txt", dtype=float)


while(True):
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # image_blurred = cv2.blur(src=frame, ksize=(10, 10))
    frame, aruco_centers = find_aruco(frame)
    # frame = find_aruco2(frame)

    frame, origin_center ,stag_centers = find_stag(frame)
    frame = cal_distance(frame, origin_center ,stag_centers)
    # print(image_blurred.shape)
    # print(stag_centers)

    


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        

    cv2.imshow("test", frame)

    if cv2.waitKey(1) == ord('q'):
        break

