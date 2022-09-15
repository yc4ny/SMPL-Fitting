import numpy as np 
from serialization import load_model
import pandas as pd
import cv2 
from tqdm import tqdm
import time

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

def smplScale(gt_joint, smpl_J_2d): 
    gt_dist = np.linalg.norm(gt_joint[15] - gt_joint[12])
    smpl_dist = np.linalg.norm(smpl_J_2d[15] - smpl_J_2d[12])

    return smpl_dist/gt_dist

if __name__ == '__main__': 

    obj = pd.read_pickle(r'3DPW/sequenceFiles/sequenceFiles/train/courtyard_arguing_00.pkl')
    smpl_params = pd.read_pickle(r'fit/output/3DPW/courtyard_arguing_00_params.pkl')
    intrinsic = obj['cam_intrinsics']
    model = load_model( 'smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    pose_params = np.array(smpl_params['pose_params'])
    pose_params = pose_params.reshape(765,24,3)
    shape_params = np.array(smpl_params['shape_params'])

    for j in tqdm(range(765)):
        gt_joint = obj['jointPositions'][0][j]
        gt_joint = np.reshape(gt_joint, (24,3))
        pose = np.reshape(pose_params[j],(72,))
        shape = shape_params[j]
        extrinsic = obj['cam_poses'][j]
        extrinsic = extrinsic[:3]
        projectionMatrix = np.matmul(intrinsic, extrinsic)
        model.pose[:] = pose
        model.betas[:] = shape
        # Vertices and Joints from SMPL parameters 
        pred_output = model()
        model.J = model.J_regressor.dot(pred_output)
        # Reproject 3d GT joint and 3d SMPL joints using the camera parameters onto the image plane
        gt_joint = reproject(projectionMatrix, gt_joint)
        smpl_J_2d = reproject(projectionMatrix,model.J)
        # Find SMPL scale and translation constant
        scale = smplScale(gt_joint, smpl_J_2d)
        smpl_J_2d = smpl_J_2d * scale
        trans_scale = [(smpl_J_2d[0][0] - gt_joint[0][0]),(smpl_J_2d[0][1] - gt_joint[0][1])] # Move to pelvis location 
        # Update SMPL joint locations to the image plane
        for i in range(smpl_J_2d.shape[0]):
            smpl_J_2d[i][0] -= trans_scale[0] 
            smpl_J_2d[i][1] -= trans_scale[1]
        # Find updated vertices using the updated SMPL joint location 
        pred_output = model()
        pred_output = reproject(projectionMatrix, pred_output)
        # Apply scale and translation to the vertices
        pred_output = pred_output * scale
        for i in range(pred_output.shape[0]):
            pred_output[i][0] -= trans_scale[0] 
            pred_output[i][1] -= trans_scale[1]
        # Print results
        img = cv2.imread('3DPW/imageFiles/imageFiles/courtyard_arguing_00/image_' + '{:05d}'.format(j)+'.jpg')

        for i in range(pred_output.shape[0]):
            joint_img = cv2.circle(img, (int( pred_output[i][0]), int(pred_output[i][1])), 1, (0,255,0),-1)
        for i in range(smpl_J_2d.shape[0]):
            joint_img = cv2.circle(img, (int(smpl_J_2d[i][0]), int( smpl_J_2d[i][1])), 5, (0,0,255),-1)
            joint_img = cv2.putText(img, str(i), (int( smpl_J_2d[i][0]), int( smpl_J_2d[i][1])),cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1, color = (255,0,0), thickness = 2)
        cv2.imwrite('output_img/'+ str(j) + '.jpg', joint_img)


