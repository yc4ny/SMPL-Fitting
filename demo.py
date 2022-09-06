import torch
import numpy as np 
from smpl.smpl_webuser.serialization import load_model


if __name__ == '__main__': 
    model = m = load_model( 'smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl' )
    model.pose = (np.random.rand((24,3)) -0.5)
    model.betas = (np.random.rand((10,)) -0.5)* 2.5

