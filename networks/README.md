## Motion VAE

### Download pre-processed dataset

1. Download the [pre-processed dataset](https://nextcloud.mpi-klsb.mpg.de/index.php/s/HWHGwZLi8xnLsRF).
2. Unzip the zip file: ```unzip EgocentricAMASS.zip```

### train the motion VAE

1. global motion VAE

```
python train.py --log_dir cnn_global_full_dataset_latent_2048_len_10_kl_0.5 --train_data_path /path/to/EgocentricAMASS --latent_dim 2048 --kl_weight 0.5 --seq_length 10 --batch_size 64 --new_dataset False --with_mo2cap2_data False --fps 25 --network cnn
```

2. local motion VAE

```
python train_local.py --log_dir mlp_local_full_dataset_latent_2048_len_10_kl_0.5 --train_data_path /path/to/EgocentricAMASS --latent_dim 2048 --kl_weight 0.5 --seq_length 10 --batch_size 64 --with_mo2cap2_data False --new_dataset False --fps 25 --network mlp
```

### sample the motion from latent space

see ```sample.py```

### interpolant two different motions

see ```interpolant.py```
