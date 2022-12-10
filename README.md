# Egocentric Pose Optimizer

The official implementation of paper: Estimating Egocentric 3D Human Pose in Global Space.

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

If you want to train the motion vae:

1. run ```mkdir AMASSDataConverter && cd AMASSDataConverter && mkdir pkl_data```
2. download the processed dataset to directory ```pkl_data``` from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/aaGCsZ4Sgz4ftge).
3. see directory ```networks```

### prepare the data for optimization
If you want to run on your own dataset,
you need to firstly preprocess the data with repo: ```MakeDataForOptimization```.

