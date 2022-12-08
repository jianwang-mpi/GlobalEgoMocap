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
from models.SeqConvVAE import ConvVAE


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
        from dataset.local_dataset import AMASSDataset
        self.seq_length = args.seq_length
        self.attention = args.attention
        
        self.amass_train_dataset = AMASSDataset(data_path=args.train_data_path, frame_num=self.seq_length, is_train=True,
                                                windows_size=args.slide_window_step, balance_distrib=args.data_balance,
                                                fps=args.fps, with_mo2cap2_data=args.with_mo2cap2_data)
        self.train_dataloader = DataLoader(dataset=self.amass_train_dataset, batch_size=args.batch_size, shuffle=True,
                                           drop_last=True, num_workers=args.num_workers)
        
        self.test_dataset = AMASSDataset(data_path=args.train_data_path, frame_num=self.seq_length, is_train=False,
                                         windows_size=args.slide_window_step, balance_distrib=args.data_balance,
                                         fps=args.fps, with_mo2cap2_data=args.with_mo2cap2_data)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False,
                                          drop_last=False, num_workers=args.num_workers)
        if args.network == 'cnn':
            self.network = ConvVAE(in_channels=45, out_channels=45, latent_dim=args.latent_dim,
                                   seq_len=self.seq_length)
        elif args.network == 'rnn':
            self.network = RecurrentVAE(in_channels=45, out_channels=45, latent_dim=args.latent_dim,
                                        seq_len=self.seq_length)
        elif args.network == 'attention':
            self.network = AttentionVAE(in_channels=45, out_channels=45, latent_dim=args.latent_dim,
                                        seq_len=self.seq_length, attention_type=self.attention)
        elif args.network == 'meva':
            self.network = MEVA(nx=45, t_total=self.seq_length, latent_size=args.latent_dim)

        elif args.network == 'vibe':
            self.network = RecurrentVAE_VIBE(in_channels=45, out_channels=45, seq_len=self.seq_length,
                                             latent_dim=args.latent_dim, hidden_size=args.latent_dim)
        elif args.network == 'mlp':
            from networks.models.MLP import MLPVAE
            self.network = MLPVAE(in_features=self.seq_length * 45, out_features=self.seq_length * 45,
                                  seq_len=self.seq_length,
                                  latent_dim=args.latent_dim)
        else:
            raise Exception('wrong network type!')
        
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
        running_recon_loss = 0
        count = 0
        for e in range(args.epoch):
            self.network.train()
            print('-----------------Epoch: {}-----------------'.format(str(e)))
            if args.new_dataset:
                for i, (relative_global_pose, local_pose, camera_matrix) in tqdm(enumerate(self.train_dataloader)):
                    self.optimizer.zero_grad()
                    # relative_global_pose = relative_global_pose.to(self.device)
                    local_pose = local_pose.to(self.device)
                    # camera_matrix = camera_matrix.to(self.device)
    
                    pose_preds, pose_input, mu, log_var = self.network(local_pose)
                    loss, recon_loss, kld_loss = self.network.loss_function(pose_preds, pose_input, mu, log_var,
                                                                            M_N=args.kl_weight * 64 / len(
                                                                                self.amass_train_dataset))
                    loss.backward()
                    self.optimizer.step()
    
                    running_loss += loss.item()
                    running_recon_loss += recon_loss.item()
                    # logs
                    if count % args.log_step == 0 and count != 0:
                        # ...log the running loss
                        self.tensorboard_writer.add_scalar('training loss', running_loss, count)
                        self.tensorboard_writer.add_scalar('training local recon loss', running_recon_loss, count)
                        print("running loss is: {}".format(running_loss))
                        print("running local recon loss is: {}".format(running_recon_loss))
                        running_loss = 0
                        running_recon_loss = 0
                    count += 1
            else:
                for i, local_pose in tqdm(enumerate(self.train_dataloader)):
                    self.optimizer.zero_grad()
                    local_pose = local_pose.to(self.device)
        
                    pose_preds, pose_input, mu, log_var = self.network(local_pose)
                    loss, recon_loss, kld_loss = self.network.loss_function(pose_preds, pose_input, mu, log_var,
                                                                            M_N=args.kl_weight * 64 / len(
                                                                                self.amass_train_dataset))
                    loss.backward()
                    self.optimizer.step()
        
                    running_loss += loss.item()
                    running_recon_loss += recon_loss.item()
                    # logs
                    if count % args.log_step == 0 and count != 0:
                        # ...log the running loss
                        self.tensorboard_writer.add_scalar('training loss', running_loss, count)
                        self.tensorboard_writer.add_scalar('training local recon loss', running_recon_loss, count)
                        print("running loss is: {}".format(running_loss))
                        print("running local recon loss is: {}".format(running_recon_loss))
                        running_loss = 0
                        running_recon_loss = 0
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
            if args.new_dataset:
                for i, (relative_global_pose, local_pose, camera_matrix) in tqdm(enumerate(self.test_dataloader)):
                    local_pose = local_pose.to(self.device)
                    pose_preds, pose_input, _, _ = self.network(local_pose)
                    pose_preds = pose_preds.cpu().numpy()
                    pose_input = pose_input.cpu().numpy()
                    pose_preds = np.reshape(pose_preds, [-1, self.seq_length, 15, 3])
                    pose_gt = np.reshape(pose_input, [-1, self.seq_length, 15, 3])
                    mpjpe_list.append(self.mpjpe(pose_preds, pose_gt))
            else:
                for i, local_pose in tqdm(enumerate(self.test_dataloader)):
                    local_pose = local_pose.to(self.device)
                    pose_preds, pose_input, _, _ = self.network(local_pose)
                    pose_preds = pose_preds.cpu().numpy()
                    pose_input = pose_input.cpu().numpy()
                    pose_preds = np.reshape(pose_preds, [-1, self.seq_length, 15, 3])
                    pose_gt = np.reshape(pose_input, [-1, self.seq_length, 15, 3])
                    mpjpe_list.append(self.mpjpe(pose_preds, pose_gt))
        
        print('MPJPE is: {}'.format(np.mean(mpjpe_list)))
        return np.mean(mpjpe_list)
    
    def mpjpe(self, pose_preds, pose_gt):
        distance = np.linalg.norm(pose_gt - pose_preds, axis=3)
        return np.mean(distance)


if __name__ == '__main__':
    train = Train()
    train.train()
