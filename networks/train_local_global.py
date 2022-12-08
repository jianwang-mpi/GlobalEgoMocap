import sys

sys.path.append('..')
import torch
import numpy as np

from config import args
import datetime
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from pprint import pprint
from utils.rigid_transform_with_scale import umeyama
# from SeqConvVAE import ConvVAE
# from SeqCCVAE import ConvVAE
from models.LocalGlobalSeqVAE import LocalGlobalSeqVAE
from models.SeqCVAE_attention import AttentionVAE
from models.RecurrentVAE import RecurrentVAE


def prepare_log_dir(log_dir=None):
    # prepare log dirs
    if args.log_dir is None and log_dir is None:
        log_dir = datetime.datetime.now().strftime('%m.%d-%H:%M:%S')
    elif log_dir is None:
        log_dir = args.log_dir
    log_dir = os.path.join('logs', log_dir)
    print('making save dir at: {}'.format(log_dir))
    os.makedirs(log_dir)
    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    os.makedirs(checkpoints_dir)
    os.makedirs(tensorboard_dir)
    return checkpoints_dir, tensorboard_dir


class Train:
    def __init__(self):
        if args.new_dataset is True:
            from dataset import AMASSDataset
        else:
            from old_dataset import AMASSDataset
        self.seq_length = args.seq_length
        self.attention = args.attention
        
        self.amass_train_dataset = AMASSDataset(data_path=args.train_data_path, frame_num=self.seq_length, is_train=True,
                                                windows_size=args.slide_window_step, balance_distrib=args.data_balance)
        self.train_dataloader = DataLoader(dataset=self.amass_train_dataset, batch_size=args.batch_size, shuffle=True,
                                           drop_last=True, num_workers=args.num_workers)
        
        self.test_dataset = AMASSDataset(data_path=args.train_data_path, frame_num=self.seq_length, is_train=False,
                                         windows_size=args.slide_window_step, balance_distrib=args.data_balance)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False,
                                          drop_last=False, num_workers=args.num_workers)
        # self.network = AttentionVAE(in_channels=45, out_channels=45, latent_dim=args.latent_dim,
        #                             seq_len=self.seq_length, attention_type=self.attention)
        self.network = LocalGlobalSeqVAE(in_channels=45, out_channels=45, local_vae_latent_dim=args.latent_dim,
                                         global_vae_latent_dim=args.latent_dim, seq_len=self.seq_length)
        # self.network = RecurrentVAE(in_channels=45, out_channels=45, latent_dim=args.latent_dim, seq_len=self.seq_length)
        
        self.optimizer = Adam(params=self.network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # self.l2_loss = torch.nn.MSELoss(reduction='elementwise_mean')
        # self.l2_loss = torch.nn.MSELoss()
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.network = self.network.to(self.device)
        
        self.checkpoint_dir, tensorboard_dir = prepare_log_dir()
        self.tensorboard_writer = SummaryWriter(tensorboard_dir)
    
    def train(self):
        print('---------------------Start Training-----------------------')
        pprint(args.__dict__)
        running_loss = 0
        running_local_recon_loss = 0
        running_global_recon_loss = 0
        count = 0
        for e in range(args.epoch):
            self.network.train()
            print('-----------------Epoch: {}-----------------'.format(str(e)))
            for i, (relative_global_pose, local_pose, camera_matrix) in tqdm(enumerate(self.train_dataloader)):
                self.optimizer.zero_grad()
                # relative_global_pose = relative_global_pose.to(self.device)
                local_pose = local_pose.to(self.device)
                camera_matrix = camera_matrix.to(self.device)

                local_pose_out, local_pose_input, mu_local, log_var_local, \
                global_pose_out, global_pose_input, mu_global, log_var_global = self.network(local_pose, camera_matrix)
                loss, other_data = \
                    self.network.loss_function(local_pose_out, local_pose_input, mu_local, log_var_local,
                                               args.kl_weight * 64 / len(self.amass_train_dataset),
                                               global_pose_out, global_pose_input, mu_global, log_var_global,
                                               args.kl_weight * 64 / len(self.amass_train_dataset))
                loss.backward()
                self.optimizer.step()
                
                
                
                running_loss += loss.item()
                running_local_recon_loss += other_data[1].item()
                running_global_recon_loss += other_data[4].item()
                # logs
                if count % args.log_step == 0 and count != 0:
                    # ...log the running loss
                    self.tensorboard_writer.add_scalar('training loss', running_loss, count)
                    self.tensorboard_writer.add_scalar('training local recon loss', running_local_recon_loss, count)
                    self.tensorboard_writer.add_scalar('training global recon loss', running_global_recon_loss, count)
                    print("running loss is: {}".format(running_loss))
                    print("running local recon loss is: {}".format(running_local_recon_loss))
                    print("running global recon loss is: {}".format(running_global_recon_loss))
                    running_loss = 0
                    running_local_recon_loss = 0
                    running_global_recon_loss = 0
                count += 1
            # log the logs
            eval_loss = self.eval()
            self.tensorboard_writer.add_scalar('eval loss', eval_loss, e)
            print('eval loss is: {}'.format(eval_loss))
            
            torch.save({
                'epoch': e + 1,
                'args': args.__dict__,
                'state_dict': self.network.state_dict(),
                'eval_result': eval_loss,
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(self.checkpoint_dir, str(e) + '.pth.tar'))
    
    def eval(self):
        print('---------------------Start Eval-----------------------')
        self.network.eval()
        mpjpe_list = []
        with torch.no_grad():
            for i, (relative_global_pose, local_pose, camera_matrix) in tqdm(enumerate(self.test_dataloader)):
                relative_global_pose = relative_global_pose.to(self.device)
                local_pose = local_pose.to(self.device)
                camera_matrix = camera_matrix.to(self.device)
    
                local_pose_preds, local_pose_input, mu_local, log_var_local, \
                global_pose_preds, global_pose_input, mu_global, log_var_global = self.network(local_pose, camera_matrix)
                
                global_pose_preds = global_pose_preds.cpu().numpy()
                global_pose_input = global_pose_input.cpu().numpy()
                relative_global_pose = np.reshape(relative_global_pose, [-1, self.seq_length, 15, 3])
                pose_gt = np.reshape(relative_global_pose, [-1, self.seq_length, 15, 3])
                mpjpe_list.append(self.mpjpe(global_pose_preds, pose_gt))
        
        print('MPJPE is: {}'.format(np.mean(mpjpe_list)))
        return np.mean(mpjpe_list)
    
    def mpjpe(self, pose_preds, pose_gt):
        distance = np.linalg.norm(pose_gt - pose_preds, axis=3)
        return np.mean(distance)


if __name__ == '__main__':
    train = Train()
    train.train()
