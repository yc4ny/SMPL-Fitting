import numpy as np 
import os 
import pandas as pd

# for img in os.listdir('3DPW/sequenceFiles/sequenceFiles/train'):
obj = pd.read_pickle(r'3DPW/sequenceFiles/train/courtyard_backpack_00.pkl')
joints = obj['jointPositions'][0]
joints = np.reshape(joints, (obj['jointPositions'][0].shape[0],24,3))
np.save('3DPW/npy/courtyard_backpack_00.npy',joints)





