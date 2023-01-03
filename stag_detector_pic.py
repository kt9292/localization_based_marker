import pyStag as stag
import cv2
import numpy as np
import math
import os

def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(-R, np.matrix(tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape(
        (3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec

def find_stag(frame):
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # set Stag Detector parameter
    det = stag.Detector(11, 7 , False)
    det.detect(grayimg)
    # get coners and ids from stag detected
    coners = det.getContours()
    ids = det.getIds()

    unique_ids = []
    unique_coners = []

    # remove duplicate ids and coners
    if len(ids) != 0:
        unique_dict = dict(zip(ids, coners))
        
        # ordered dict
        unique_dict = sorted(unique_dict.items())
        for id, coner in unique_dict:
            unique_ids.append(id)
            unique_coners.append(coner)

    centers = []
    i = 0
    origin_center = []
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
        # rvec : rotation vector , tvec : trasformation vector
        # estimatePoseSingleMarkers(marker edge point, markerLength, intrinsic matrix, distortion matrix)
        rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(bbx_for_axis, 0.08, intrinsic_matrix, distortion_matrix)

        tvec_added_id = np.append(tvec,id).reshape(1,1,4)

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
            origin_center = center
            origin_center.append(id)
        else :
            center.append(id)
            centers.append(center)

        i += 1

    return frame, origin_center ,centers, tvecs, rvecs

def cal_distance(frame, origin_center ,centers, tvecs, rvecs):
    t = np.array(tvecs)
    r = np.array(rvecs)
    
    for i in range(0, len(t)):
        if(int(t[i][0][0][3]) == 0):
            # tvec0, rvec0 : transformation vectors of id 0
            tvec0 = t[i][:,:,:3]
            rvec0 = r[i]

            for center, j in zip(centers, range(i+1, len(t))):
                # tvec1, rvec1 : transformation vectors of id n    
                tvec1 = t[j][:,:,:3]
                rvec1 = r[j]

                tvec0_x = tvec0[0][0][0]
                tvec0_y = tvec0[0][0][1]
                tvec0_z = tvec0[0][0][2]
                tvec1_x = tvec1[0][0][0]
                tvec1_y = tvec1[0][0][1]
                tvec1_z = tvec1[0][0][2]

                # calculate distacne between id:0 with id:n
                dist1 = math.sqrt(pow((tvec0_x-tvec1_x),2)+pow((tvec0_y-tvec1_y),2)+pow((tvec0_z-tvec1_z),2))
                distance= f'{dist1*100:.2f}cm'
                
                # calculate relative position of marker from origin marker
                composedRvec, composedTvec = relativePosition(rvec0, tvec0, rvec1, tvec1)
                relative_position = []
                for p in composedTvec:
                    relative_position.append(p[0])
                relative_position_str = f'({relative_position[0]:.3f}, {relative_position[1]:.3f}, {relative_position[2]:.3f})'
        
                    
                frame = cv2.putText(frame, distance,(int(abs(center[0]+origin_center[0])/2), int(abs(center[1] + origin_center[1])/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2, cv2.LINE_4)
                frame = cv2.putText(frame, relative_position_str, (int(center[0]), int(center[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2, cv2.LINE_4)
                frame = cv2.line(frame, (origin_center[0],origin_center[1]), (center[0], center[1]), (255,255,0), 2)
        else:
            pass

    return frame



# camera config
intrinsic_matrix = np.loadtxt("./config/intrinsic_matrix_logitech.txt", dtype=float)
distortion_matrix = np.loadtxt("./config/distortion_matrix_logitech.txt", dtype=float)

# file load
file_path = "./sample"
file_list = os.listdir(file_path)
imgs_path = []
output_path = "./result"

for f in file_list:
    img = f'{file_path}/{f}'
    imgs_path.append(img)

for i in imgs_path:
    print(i)
    frame = cv2.imread(i, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame, origin_center ,stag_centers, tvecs, rvecs = find_stag(frame)
    frame = cal_distance(frame, origin_center ,stag_centers, tvecs, rvecs) 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_name = f'{i[9:-4]}_result.jpg'
    print(f'result saved at {output_path}/{output_name}')
    cv2.imwrite(f'{output_path}/{output_name}', frame)

