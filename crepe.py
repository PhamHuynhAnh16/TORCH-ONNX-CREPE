import os, sys, torch, functools
sys.path.append(os.getcwd())
from .core import PITCH_BINS

class Crepe(torch.nn.Module):
    def __init__(self, model='full'):
        super().__init__()
        if model == 'full':
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif model == 'large':
            in_channels = [1, 768, 96, 96, 96, 192]
            out_channels = [768, 96, 96, 96, 192, 384]
            self.in_features = 1536
        elif model == 'medium':
            in_channels = [1, 512, 64, 64, 64, 128]
            out_channels = [512, 64, 64, 64, 128, 256]
            self.in_features = 1024
        elif model == 'small':
            in_channels = [1, 256, 32, 32, 32, 64]
            out_channels = [256, 32, 32, 32, 64, 128]
            self.in_features = 512
        elif model == 'tiny':
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256

        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        batch_norm_fn = functools.partial(torch.nn.BatchNorm2d, eps=0.0010000000474974513, momentum=0.0)

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_sizes[0], stride=strides[0])
        self.conv1_BN = batch_norm_fn(num_features=out_channels[0])
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_sizes[1], stride=strides[1])
        self.conv2_BN = batch_norm_fn(num_features=out_channels[1])
        self.conv3 = torch.nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_sizes[2], stride=strides[2])
        self.conv3_BN = batch_norm_fn(num_features=out_channels[2])
        self.conv4 = torch.nn.Conv2d(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=kernel_sizes[3], stride=strides[3])
        self.conv4_BN = batch_norm_fn(num_features=out_channels[3])
        self.conv5 = torch.nn.Conv2d(in_channels=in_channels[4], out_channels=out_channels[4], kernel_size=kernel_sizes[4], stride=strides[4])
        self.conv5_BN = batch_norm_fn(num_features=out_channels[4])
        self.conv6 = torch.nn.Conv2d(in_channels=in_channels[5], out_channels=out_channels[5], kernel_size=kernel_sizes[5], stride=strides[5])
        self.conv6_BN = batch_norm_fn(num_features=out_channels[5])
        
        self.classifier = torch.nn.Linear(in_features=self.in_features, out_features=PITCH_BINS)

    def forward(self, x, embed=False):
        x = self.embed(x)

        if embed: return x
        return torch.sigmoid(self.classifier(self.layer(x, self.conv6, self.conv6_BN).permute(0, 2, 1, 3).reshape(-1, self.in_features)))

    def embed(self, x):
        x = x[:, None, :, None]
        return self.layer(self.layer(self.layer(self.layer(self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254)), self.conv2, self.conv2_BN), self.conv3, self.conv3_BN), self.conv4, self.conv4_BN), self.conv5, self.conv5_BN)

    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        return torch.nn.functional.max_pool2d(batch_norm(torch.nn.functional.relu(conv(torch.nn.functional.pad(x, padding)))), (2, 1), (2, 1))