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


### Run the optimization for our dataset

```
python optimize_whole_sequence.py --data_path data/jian3
python optimize_whole_sequence.py --data_path data/studio-jian1
python optimize_whole_sequence.py --data_path data/studio-jian2
python optimize_whole_sequence.py --data_path data/studio-lingjie1
python optimize_whole_sequence.py --data_path data/studio-lingjie2
```

### Train the motion vae

If you want to train the motion vae, please see directory ```networks```


### prepare the data for optimization
If you want to run on your own dataset,
you need to firstly preprocess the data with repo: ```MakeDataForOptimization```.

