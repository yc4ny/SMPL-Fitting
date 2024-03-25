# Fitting SMPL Parameters to 3D Keypoints
Code repository for the paper:
**Reconstructing Hands in 3D with Transformers**

![teaser](assets/teaser.jpg)

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

You also need to download the trained models:
```bash
bash fetch_demo_data.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section.  We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.

## Demo
```bash
python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
```

## Training
First, download the training data to `./hamer_training_data/` by running:
```
bash fetch_training_data.sh
```

Then you can start training using the following command:
```
python train.py exp_name=hamer data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```
Checkpoints and logs will be saved to `./logs/`.

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [4DHumans](https://github.com/shubham-goel/4D-Humans)
- [SLAHMR](https://github.com/vye16/slahmr)
- [ProHMR](https://github.com/nkolot/ProHMR)
- [SPIN](https://github.com/nkolot/SPIN)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)
- [HMR](https://github.com/akanazawa/hmr)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [Detectron2](https://github.com/facebookresearch/detectron2)

Additionally, we thank [StabilityAI](https://stability.ai/) for a generous compute grant that enabled this work.

## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}
```