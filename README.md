# Egocentric Pose Optimizer

The official implementation of paper: 

### Estimating Egocentric 3D Human Pose in Global Space.

[[project page]](https://vcai.mpi-inf.mpg.de/projects/globalegomocap/)

If you find this repository useful, please cite:

```
@InProceedings{Wang_2021_ICCV,
    author    = {Wang, Jian and Liu, Lingjie and Xu, Weipeng and Sarkar, Kripasindhu and Theobalt, Christian},
    title     = {Estimating Egocentric 3D Human Pose in Global Space},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11500-11509}
}
```

### Optimize the motion sequences in proposed dataset

1. Install pytorch 1.4+ with cuda support.
2. Run ```mkdir networks/logs``` and download the trained VAE model into directory ```networks/logs``` from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/ibBB7TbEsWQrMJa).
3. Run ```mkdir data``` and download the processed test sequences into directory ```data``` from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/kLNeAdbJzmSYKsZ).
4. Run the test on the sequences:
```
python optimize_whole_sequence.py --data_path data/jian3
python optimize_whole_sequence.py --data_path data/studio-jian1
python optimize_whole_sequence.py --data_path data/studio-jian2
python optimize_whole_sequence.py --data_path data/studio-lingjie1
python optimize_whole_sequence.py --data_path data/studio-lingjie2
```

### Train the motion vae

If you want to train the motion VAE, See directory ```networks```

### prepare the data for optimization
If you want to run on your own dataset,
you need to firstly preprocess the data following the scripts in directory: ```MakeDataForOptimization```.
Basically you need to prepare the following data:
1. Predicted human body heatmap in egocentric view.
2. Predicted human body joint depths in egocentric view.
3. Human body ground truth (for calculating MPJPEs).
4. Camera pose sequence from the OpenVSLAM method.

All of these data are combined into a single pickle file for each sequence.

