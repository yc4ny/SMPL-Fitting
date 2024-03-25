# Fitting SMPL Parameters to 3D Keypoints

![teaser](assets/teaser.png)

## Installation
First you need to clone the repo:
```
git clone https://github.com/yc4ny/SMPL-Fitting.git
cd SMPL-Fitting
```
This code has been tested on **Python 3.7**, **Pytorch 1.7.1**, **CUDA 11.0**.
We recommend creating a virtual environment for this repository. You can use conda:
```
conda create -n smplfitting python==3.7
```

Then, you need to install Pytorch according to your CUDA version and GPU requriements. This is for CUDA 11.0, but you can adapt accordingly: 
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Then, you can install the rest of the dependencies with: 
```
pip install -r requriements.txt
```

## Preprocessing Files


## Demo
```bash
python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
```



## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [SMPLPytorch](https://github.com/gulvarol/smplpytorch)