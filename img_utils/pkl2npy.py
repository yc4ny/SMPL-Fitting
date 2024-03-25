import numpy as np 
import pickle 


def main():
    with open('data/3DPW/sequenceFiles/train/outdoors_freestyle_00.pkl','rb') as f:
        # 3DPW PKL File encoded in latin1 
        data = pickle.load(f, encoding= 'latin1')

        shape = (data['jointPositions'][0].shape[0], 24,3)
        data = np.reshape(data['jointPositions'][0], shape)
        print(data.shape)
        np.save("outdoors_freestyle", data, allow_pickle=True, fix_imports= True)

if __name__ == "__main__":
    main()