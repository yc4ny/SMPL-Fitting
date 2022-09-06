import torch
import numpy as np 
from serialization import load_model
import pandas as pd
import cv2 

def homogeneous_2d(point_2d): 
    point_2d[0] = point_2d[0]/point_2d[2]
    point_2d[1] = point_2d[1]/point_2d[2]
    point_2d[2] = point_2d[2]/point_2d[2]
    return point_2d

def reproject(projectionMatrix, point_3d):
    points_2d = np.empty((point_3d.shape[0],3))
    ones = np.ones((point_3d.shape[0],1))
    point_3d = np.hstack((point_3d,ones))

    for i in range(point_3d.shape[0]):
        reprojected = np.matmul(projectionMatrix,point_3d[i])
        homogeneous_2d(reprojected)
        points_2d[i][0] = reprojected[0]
        points_2d[i][1] = reprojected[1]
        points_2d[i][2] = reprojected[2]

    return points_2d[:,:2]
if __name__ == '__main__': 

    obj = pd.read_pickle(r'3DPW/sequenceFiles/sequenceFiles/train/courtyard_arguing_00.pkl')
    joint = obj['jointPositions'][0][0]
    joint = np.reshape(joint, (24,3))
    intrinsic = obj['cam_intrinsics']
    extrinsic = obj['cam_poses'][0]
    extrinsic = extrinsic[:3]
    projectionMatrix = np.matmul(intrinsic, extrinsic)
    

    model = m = load_model( 'smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl' )
    model.pose[:] = obj['poses'][0][0]
    model.betas[:] = obj['betas'][0][0]
    point_2d = reproject(projectionMatrix,joint)
    # model.r: Vertices
    # model.f: Faces
    # model.J: 3D Joints
    img = cv2.imread('3DPW/imageFiles/imageFiles/courtyard_arguing_00/image_00000.jpg')

    for i in range(24):
        joint_img = cv2.circle(img, (int(point_2d[i][0]), int(point_2d[i][1])), 10, (0,0,255),-1)

    cv2.imwrite('test.jpg', joint_img)


