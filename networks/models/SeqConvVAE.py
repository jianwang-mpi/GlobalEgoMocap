import torch
from models.BaseVAE import BaseVAE
# from networks.BaseVAE import BaseVAE
from torch import nn
from torch.nn import functional as F
from copy import deepcopy


class ConvVAE(BaseVAE):
    kinematic_parents = [0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_dim,
                 seq_len,
                 hidden_dims=None,
                 with_bone_length=False,
                 **kwargs) -> None:
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.with_bone_length = with_bone_length
        
        modules = []
        self.encoder_hidden_dims = deepcopy(hidden_dims)
        if hidden_dims is None:
            hidden_dims = [64, 64, 128, 256, 512]
        
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.encoder_out_dim = hidden_dims[-1] * self.seq_len
        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_out_dim, latent_dim)
        
        if self.with_bone_length is True:
            self.bone_length_encode_layer = nn.Sequential(
                nn.Linear(in_features=self.seq_len * 15, out_features=512),
                nn.BatchNorm1d(num_features=512),
                nn.LeakyReLU()
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(in_features=512 + hidden_dims[-1] * self.seq_len, out_features=hidden_dims[-1] * self.seq_len),
                nn.BatchNorm1d(num_features=hidden_dims[-1] * self.seq_len),
                nn.LeakyReLU()
            )
        
        # Build Decoder
        modules = []
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.seq_len)
        
        hidden_dims.reverse()
        self.decoder_hidden_dims = deepcopy(hidden_dims)
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       output_padding=0),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               output_padding=0),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims[-1], out_channels=self.out_channels,
                      kernel_size=3, padding=1))
        
        
        
    
    def encode(self, pose_input: torch.Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param pose_input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(pose_input)
        result = torch.flatten(result, start_dim=1)
        
        if self.with_bone_length is True:
            bone_length_encoding = self.bone_length_encoding(pose_input)
            result = torch.cat([result, bone_length_encoding], dim=1)
            result = self.fusion_layer(result)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return [mu, log_var]
    
    def decode(self, z: torch.Tensor):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.decoder_hidden_dims[0], self.seq_len)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def decode_to_bodypose(self, z: torch.Tensor):
        # result: (n_batch, 45, seq_len)
        # output: (n_batch, seq_len, 15, 3)
        result = self.decoder_input(z)
        result = result.view(-1, self.decoder_hidden_dims[0], self.seq_len)
        result = self.decoder(result)
        result = self.final_layer(result)
        assert result.shape[1] == 45
        output = result.permute([0, 2, 1]).view((-1, self.seq_len, 15, 3))
        return output
    
    def calculate_bone_length(self, skeleton: torch.Tensor):
        # skeleton shape: (seq_len, 15 * 3)
        skeleton = skeleton.view((-1, self.seq_len, 15, 3))
        bone_array = skeleton - skeleton[:, :, self.kinematic_parents, :]
        bone_length = torch.norm(bone_array, dim=3)
        return bone_length
    
    def bone_length_encoding(self, pose_input):
        pose_input = pose_input.view(-1, self.seq_len, 15, 3)
        # calculate the bone length
        bone_length = self.calculate_bone_length(pose_input)
        bone_length = bone_length.view(-1, self.seq_len * 15)
        bone_length_encoding = self.bone_length_encode_layer(bone_length)
        return bone_length_encoding
        
        
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, pose_input: torch.Tensor, **kwargs):
        """
        :param pose_input: shape: n_batch, seq_length, 15*3
        :param kwargs:
        :return:
        """
        pose_input_permuted = pose_input.permute((0, 2, 1)).contiguous()
        mu, log_var = self.encode(pose_input_permuted)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        out = out.permute((0, 2, 1))
        return out, pose_input, mu, log_var
    
    def get_latent_space(self, pose_input):
        pose_input_permuted = pose_input.permute((0, 2, 1)).contiguous()
        mu, log_var = self.encode(pose_input_permuted)
        std = torch.exp(0.5 * log_var)
        z = self.reparameterize(mu, log_var)
        return mu, std, z
    
    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        if 'M_N' in kwargs:
            kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
            # print(kld_weight)
            recons_loss = F.mse_loss(recons, input)

            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        else:
            kld_weight = kwargs['kl_weight']
            recons_loss = F.mse_loss(recons, input, reduction='sum')

            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        loss = recons_loss + kld_weight * kld_loss
        return loss, recons_loss, kld_loss
    
    def sample(self, num_samples, current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: Number of samples
        :param current_device:  Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        
        z = z.to(current_device)
        
        samples = self.decode(z)
        samples = samples.permute((0, 2, 1))
        return samples

    
    def generate(self, x: torch.Tensor, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        
        return self.forward(x)[0]

if __name__ == '__main__':
    convVAE = ConvVAE(in_channels=45, out_channels=45, latent_dim=128, seq_len=5, with_bone_length=True)
    pose = torch.zeros(4, 5, 15 * 3)
    # pose = pose.permute((0, 2, 1)).contiguous()
    result, pose_in, _, _ = convVAE(pose)
    print(result.shape)
    # print(pose_in)
    
    # sampled_result = convVAE.sample(20, torch.device('cpu'))
    #
    # print(sampled_result.shape)