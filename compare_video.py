import pyStag as stag
import cv2
import numpy as np
import time
from utils.bg import bg 

def find_aruco(frame, marker_size=6, total_markers=250, draw=True):
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    # key = getattr(cv2.aruco, "DICT_ARUCO_ORIGINAL")
    aruco_dict = cv2.aruco.Dictionary_get(key)
    aruco_param = cv2.aruco.DetectorParameters_create()
    bbox, ids, _ = cv2.aruco.detectMarkers(grayimg, aruco_dict, parameters=aruco_param)

    centers = []
    # if len(bbox) != 0:
    #     i=0
    #     for bbx in bbox:
    #         print(f'bbx {i}:{bbx}\n')
    #         i+=1
    
    if len(bbox) != 0:
        for bbx, id in zip(bbox,ids):
            top_left, top_right, bottom_right, bottom_left = bbx[0]
            
            top_left = [int (x) for x in top_left]
            top_right = [int (x) for x in top_right]
            bottom_right = [int (x) for x in bottom_right]
            bottom_left = [int (x) for x in bottom_left]
            center = [int((top_left[0] + bottom_right[0]) / 2.0), int((top_left[1] + bottom_right[1]) / 2.0)]
            centers.append(center)            

            # rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(bbx, 0.129, intrinsic_matrix, distortion_matrix)
            # cv2.drawFrameAxes(frame, intrinsic_matrix, distortion_matrix, rvec, tvec, 0.05)

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

now = time
path = "./sample"
filename = "230223_aruco_test.mp4"
# camera config

cam = cv2.VideoCapture(f'{path}/{filename}')

w = round(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = cam.get(cv2.CAP_PROP_FPS)

result = cv2.VideoWriter(f'{filename.split(".")[0]}_result.mp4',fourcc,fps, (w, h))



while(True):
    _, frame = cam.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # frame = cv2.blur(src=frame, ksize=(10, 10))
    frame, aruco_centers = find_aruco(frame)
    # frame = find_aruco2(frame)

    frame, origin_center ,stag_centers = find_stag(frame)
    # print(image_blurred.shape)  

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     
    
    # Capture Video
    result.write(frame)   
    
    # Take a pic
    # if cv2.waitKey(1) == ord('a'):
    #     cv2.imwrite(f'./result/{now.time()}.jpg', frame)
    #     print(f'save {now.time()}.jpg')
    
    if cv2.waitKey(1) == ord('q'):
        break

    cv2.imshow("test", frame)

  


