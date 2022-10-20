# Optimizing SMPL parameters
This is my implementation of optimizing SMPL parameters from ground truth 3D joints.<br/>
For testing, I've used the <a href="https://virtualhumans.mpi-inf.mpg.de/3DPW/">3D Poses in the Wild</a> dataset for obtaining ground truth 3D joints.<br/>
<p align= "center">
<img src="img/fitting_initial.jpg" width="290" height="500" />
<img src="img/smpl_joint_vertices.jpg" width="290" height="500" style="float:center"/> 
</p>
> <strong>Left: Before Optimization, Right: After Optimization.</strong> <br/>
> Optimized Joint Locations are marked in red points. <br/>
> SMPL Joint Numbers are in blue text.<br/> 
> Green mesh is consisted of 6930 vertices obtained from the SMPL model<br/> 
<br/>

## Environment Setup
> Note: This code was developed on Ubuntu 20.04 with Python 3.7. Later versions should work, but have not been tested.
Create and activate a virtual environment to work in, e.g. using Conda:

```
conda create -n smpl_fitting python=3.7
conda activate smpl_fitting
```
Install the remaining requirements with pip:
```
pip install -r requirements.txt
```
You must also have _ffmpeg_ installed on your system to save visualizations. <br/><br/>

### Download SMPL models
Download [SMPL Female and Male](https://smpl.is.tue.mpg.de/) and [SMPL Netural](https://smplify.is.tue.mpg.de/), and rename the files and extract them to `<current directory>/smpl/models/`, eventually, the `<current directory>/smpl` folder should have the following structure:
   ```
   smpl
    └-- models
    	└-- basicModel_f_lbs_10_207_0_v1.0.0.pkl
		└-- basicmodel_m_lbs_10_207_0_v1.0.0.pkl
		└-- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
   ```
### Install the smplpytorch package
You need this package to obtain the `SMPL_Layer` used for optimization.
```
pip install smplpytorch
```

### Optimization
Before running optimization you must convert groundtruth 3d joints into the required format.<br/>
If using 3DPW dataset, just change the directory of the sequence files of your choice in the `pkl_to_npy` file. <br/>
After run:
```
python pkl_to_npy.py
```
Once you've obtained the `.npy` file containing the groundtruth 3d joints, run optimization. <br/>
For the 3DPW dataset: 
```
python fit/tools/main.py --dataset_name 3DPW --dataset_path 3DPW/npy
```





