import pyStag as stag
import cv2
import numpy as np
import time
from utils.bg import bg 
import pandas as pd
import os

def check_key_aruco(dict):
    if 'A_O_0' in dict:
        pass
    else:
        dict['A_O_0'] = 0

    if 'A_O_1' in dict:
        pass
    else:
        dict['A_O_1'] = 0

    if 'A_O_2' in dict:
        pass
    else:
        dict['A_O_2'] = 0

    if 'A_R_0' in dict:
        pass
    else:
        dict['A_R_0'] = 0

    if 'A_R_1' in dict:
        pass
    else:
        dict['A_R_1'] = 0

    if 'A_R_2' in dict:
        pass
    else:
        dict['A_R_2'] = 0


    return dict


def check_key_Stag(dict):
    if 'S_O_0' in dict:
        pass
    else:
        dict['S_O_0'] = 0

    if 'S_O_1' in dict:
        pass
    else:
        dict['S_O_1'] = 0

    if 'S_O_2' in dict:
        pass
    else:
        dict['S_O_2'] = 0

    if 'S_R_0' in dict:
        pass
    else:
        dict['S_R_0'] = 0

    if 'S_R_1' in dict:
        pass
    else:
        dict['S_R_1'] = 0

    if 'S_R_2' in dict:
        pass
    else:
        dict['S_R_2'] = 0

    return dict


def find_aruco(frame, marker_size=4, total_markers=250, draw=True):
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # key = getattr(cv2.aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    key = getattr(cv2.aruco, "DICT_ARUCO_ORIGINAL")
    aruco_dict = cv2.aruco.Dictionary_get(key)
    aruco_param = cv2.aruco.DetectorParameters_create()
    bbox, ids, _ = cv2.aruco.detectMarkers(grayimg, aruco_dict, parameters=aruco_param)

    centers = []
    detect = {}
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
            # frame = cv2.putText(frame, "{}".format(id[0]),top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,225), 2)
            if center[0] < 1280:
                if id[0] == 0 or id[0] == 1 or id[0] ==2:
                    detect[f'A_O_{id[0]}'] = 1
                else:
                    detect[f'A_O_error'] = 1
            else:
                if id[0] == 0 or id[0] == 1 or id[0] ==2:
                    detect[f'A_R_{id[0]}'] = 1
                else:
                    detect[f'A_R_error'] = 1

    # check dict
    detect = check_key_aruco(detect)

    # ordered dict
    detect = sorted(detect.items())
    detect2 = {}
    for k, v in detect:
        detect2[f'{k}'] = v
    
    return frame, centers, detect2



def find_stag(frame):
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det = stag.Detector(11, 7 , False)
    det.detect(grayimg)
    coners = det.getContours()
    ids = det.getIds()

    centers = []
    i = 0
    O_center = []
    detect = {}

    for bbx, id in zip(coners, ids):
        top_left, top_right, bottom_right, bottom_left = bbx
        
        top_left = [int (x) for x in top_left]
        top_right = [int (x) for x in top_right]
        bottom_right = [int (x) for x in bottom_right]
        bottom_left = [int (x) for x in bottom_left]
        center = [int((top_left[0] + bottom_right[0]) / 2.0), int((top_left[1] + bottom_right[1]) / 2.0)]

        bbx_for_axis = np.rint(np.array(bbx).reshape(1,4,2)) 

        # rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(bbx_for_axis, 0.129, intrinsic_matrix, distortion_matrix)
        # cv2.drawFrameAxes(frame, intrinsic_matrix, distortion_matrix, rvec, tvec, 0.04)

        frame = cv2.line(frame,top_left, top_right,(255,255,0),1)
        frame = cv2.line(frame,top_right,bottom_right,(255,225,0),1)
        frame = cv2.line(frame,bottom_right,bottom_left,(255,0,225),1)
        frame = cv2.line(frame,bottom_left, top_left,(255,0,225),1)
        frame = cv2.line(frame, center, center, (255,0,0),3)
        # frame = cv2.putText(frame, "{}".format(id),top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,225), 2)
        
        if center[0] < 1280:
            if id == 0 or id == 1 or id ==2:
                detect[f'S_O_{id}'] = 1
            else:
                detect[f'S_O_error'] = 1
        else:
            if id == 0 or id == 1 or id ==2:
                detect[f'S_R_{id}'] = 1
            else:
                detect[f'S_R_error'] = 1

        if i % 4 == 3:
            if id == 0:
                O_center = center
                O_center.append(id)
            else :
                center.append(id)
                centers.append(center)

        i += 1
    
    
    # check dict
    detect = check_key_Stag(detect)
    
    # ordered dict
    detect = sorted(detect.items())
    detect2 = {}
    for k, v in detect:
        detect2[f'{k}'] = v

    return frame, O_center ,centers , detect2

model_name = 'UDnet'
img_dir = f'../data/sample/{model_name}_sample3'
img_path = os.listdir(img_dir)

result_dir = f'../data/result/{model_name}'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


column_name = ['modelName', 'fileName', 'maskType', 'blendingWeight']
df = pd.DataFrame()
df = df.reindex(columns = column_name)

column_name2 = ['A_O_0', 'A_O_1','A_O_2','A_R_0','A_R_1','A_R_2']
df_aruco = pd.DataFrame()
df_aruco = df_aruco.reindex(columns = column_name2)

column_name3 = ['S_O_0', 'S_O_1','S_O_2','S_R_0','S_R_1','S_R_2']
df_Stag = pd.DataFrame()
df_Stag = df_Stag.reindex(columns = column_name3)


i = 0
for img_name in img_path:
    img = cv2.imread(f'{img_dir}/{img_name}')
    img_name2 = img_name.split('.')[0].split('_')
    
    print(img_name2)
    df_tmp = pd.DataFrame({'modelName':img_name2[4], 'fileName':img_name2[3], 'maskType':img_name2[2], 'blendingWeight':img_name2[0]}, index=[0])
    df = pd.concat([df, df_tmp], axis=0, ignore_index = True)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img, aruco_centers, detect_aruco = find_aruco(img)
    df_aruco_tmp = pd.DataFrame(detect_aruco, index=[0])
    # print(df_aruco_tmp)
    df_aruco = pd.concat([df_aruco, df_aruco_tmp], axis=0, ignore_index=True)

    img, origin_center , stag_centers, detect_Stag = find_stag(img)
    df_Stag_tmp = pd.DataFrame(detect_Stag, index=[0])
    # print(df_Stag_tmp)
    df_Stag = pd.concat([df_Stag, df_Stag_tmp], axis=0, ignore_index=True)\

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{result_dir}/{img_name}', img)


print(df)
print(df_aruco)
print(df_Stag)

df_result = pd.concat([df, df_aruco, df_Stag], axis=1)
print(df_result)
df_result.to_csv(f'{result_dir}/result_{model_name}.csv', sep=',')


  


