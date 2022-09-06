import pandas as pd
import numpy as np 
import cv2 

def homogeneous_2d(point_2d): 
    point_2d[0] = point_2d[0]/point_2d[2]
    point_2d[1] = point_2d[1]/point_2d[2]
    point_2d[2] = point_2d[2]/point_2d[2]
    return point_2d

def reproject(cameraParams, point_3d):
    point_2d = np.empty((point_3d.shape[0],2))
    for i in range(point_3d.shape[0]):
        reprojected = np.matmul(cameraParams, point_3d[i])
        reprojected = homogeneous_2d(reprojected)
        point_2d[i][0] = reprojected[0]
        point_2d[i][1] = reprojected[1]
    return point_2d


obj = pd.read_pickle(r'3DPW/sequenceFiles/sequenceFiles/train/courtyard_arguing_00.pkl')
print(obj)

joint = obj['jointPositions'][0][0]
joint = np.reshape(joint, (24,3))
intrinsic = obj['cam_intrinsics']
extrinsic = obj['cam_poses'][0]

point_2d = reproject(intrinsic,joint)

img = cv2.imread('3DPW/imageFiles/imageFiles/courtyard_arguing_00/image_00000.jpg')

for i in range(24):
    joint_img = cv2.circle(img, (int(point_2d[i][0]), int(point_2d[i][1])), 50, (0,0,255),-1)

cv2.imwrite('test.jpg', joint_img)
