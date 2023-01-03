import cv2
import numpy as np
import os
import glob

# define dimesion of checkerboard
CHECKERBOARD = (8,6) # the number of inner coners each columns and rows
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# create vector used to save 3D point about cheakerboard
objpoints = []
# create vector used to save 2D point about cheakerboard
imgpoints = [] 
# define world coordinate of 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
# images load for calibration
images = glob.glob('./images/*.jpg')

file_path = "./images"
file_list = os.listdir(file_path)
imgs_path = []
output_path = "./calibration_result"

for f in file_list:
    img = f'{file_path}/{f}'
    imgs_path.append(img)

for fname in imgs_path:
    img = cv2.imread(fname)
    # convert to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # find coner on checkerboard
    # if find the number of checkerboard coners, return ret = true
    ret, corners = cv2.findChessboardCorners(gray,
                                             CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # if find the number of checkerboard coners,
    # fine tuning pixel coordinate -> draw calibration result
    if ret == True:
        objpoints.append(objp)
        # fine tuning pixel coordinate using 2D point 
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)
        # draw calibration result
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    cv2.imshow('img',img)
    output_name = f'{fname[9:-4]}_result.jpg'
    cv2.imwrite(f'{output_path}/{output_name}', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
h,w = img.shape[:2] # 480, 640
# camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n") # intrinsic matrix
print(mtx)
output_path = './config'
file_name1 = 'intrinsic_matrix_logitech.txt'
result1 = np.savetxt(f'{output_path}/{file_name1}',mtx)


print("dist : \n") # Lens distortion coefficients
print(dist)
file_name2 = 'distortion_matrix_logitech.txt'
result2 = np.savetxt(f'{output_path}/{file_name2}',dist)
