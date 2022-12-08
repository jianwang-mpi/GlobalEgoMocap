import argparse

parser = argparse.ArgumentParser(description='Estimate the ego-motion from single images')

parser.add_argument('--train_data_path', default='/home/jianwang/ScanNet/static00/EgocentricAMASSPytorch/relative_global_pose_20_25.h5',
                    type=str, help='train data dir')
#
parser.add_argument('--test_data_path', default='/home/jianwang/ScanNet/static00/EgocentricAMASSPytorch/relative_global_pose_20_25.h5',
                    type=str, help='test data dir')

parser.add_argument('--network', type=str, default='cnn', required=False, help='network type')

parser.add_argument('--attention', type=str, default=None, required=False, help='attention type')

parser.add_argument('--latent_dim', type=int, required=True, help='vae latent dimension')

parser.add_argument('--with_mo2cap2_data', type=lambda x: (str(x).lower() == 'true'), required=True, default=False, help='use mo2cap2 dataset')

parser.add_argument('--new_dataset', type=lambda x: (str(x).lower() == 'true'), required=True, default=True, help='use new dataset')

parser.add_argument('--data_balance', type=lambda x: (str(x).lower() == 'true'), required=False, default=False, help='balance walking data')

parser.add_argument('--slide_window_step', default=1, type=int, required=False, help='size of sample window')

parser.add_argument('--seq_length', type=int, required=True, help='length of sequence')

parser.add_argument('--fps', type=int, required=True, default=25, help='fps')

parser.add_argument('--kl_weight', default=0.25, type=float, required=True, help='kl weight')

parser.add_argument('--epoch', default=20, type=int, help='number of total epochs to run')

parser.add_argument('--batch_size', default=64, type=int, help='batch size on every gpu')

parser.add_argument('--num_workers', default=8, type=int, help='number of workers loading data')

parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')

parser.add_argument('--weight_decay', default=0, type=float, help='weight decay (default: 1e-4)')

parser.add_argument('--log_dir', default=None, type=str, help='logging directory')



parser.add_argument('--log_prefix', default=None, type=str, help='logging prefix')

parser.add_argument('--log_step', default=100, type=int, help='logging step')

parser.add_argument('--save_step', default=2000, type=int, help='saving step')

args = parser.parse_args()