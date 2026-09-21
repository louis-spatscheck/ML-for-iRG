"""Convolutional networks for the inverse RG transformation of 2D Ising configurations.

All models map a coarse configuration of size L x L to a fine one of size
2L x 2L. Convolutions use circular padding, i.e. periodic boundary
conditions. Inputs and targets are shifted by +1 so that spins take the values
{0, 2} instead of {-1, +1} (thesis Sec. 7.2); this keeps the ReLU activations
effective.

Layer names are unchanged from the original training scripts, so saved
``state_dict`` checkpoints can be loaded directly.
"""
import torch
from torch import nn

__all__ = ["ShallowCNN", "DeepCNN", "UNet"]


class ShallowCNN(nn.Module):
    """Shallow model: one transposed convolution followed by one 3x3 convolution.

    Doubles the lattice size, (N, 1, L, L) -> (N, 1, 2L, 2L). 2,049 parameters
    (thesis Table A.1, "Simple"). Too small to learn the inverse RG
    transformation (thesis Sec. 7.2).
    """

    def __init__(self):
        super().__init__()

        self.Tconv = nn.Sequential(
            nn.ConvTranspose2d(1, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),  # batch normalization
            #nn.Dropout2d(0.2),
            nn.ReLU(inplace = True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1,padding_mode="circular")
        )

    def forward(self,x):

        x = self.Tconv(x)
        x = self.final_conv(x)

        return x


class DeepCNN(nn.Module):
    """Deep model: three residual blocks around a single upsampling step.

    Doubles the lattice size, (N, 1, L, L) -> (N, 1, 2L, 2L). 3,221,249
    parameters (thesis Table A.1, "Deep"). Note that the first residual
    connection adds the 1-channel input to all 256 feature channels by
    broadcasting.
    """

    def __init__(self):
        super().__init__()

        self.Tconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),  # batch normalization
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace = True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # batch normalization
            nn.ReLU(inplace = True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # batch normalization

        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # batch normalization
            nn.ReLU(inplace = True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # batch normalization
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # batch normalization
            nn.ReLU(inplace = True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # batch normalization
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, stride=1,padding=1,padding_mode="circular")
        )

    def forward(self,x):

        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)

        x = nn.functional.relu(x + shortcut)

        x = self.Tconv(x)

        shortcut = x
        x = self.conv3(x)
        x = self.conv4(x)

        x = nn.functional.relu(x + shortcut)

        shortcut = x
        x = self.conv5(x)
        x = self.conv6(x)

        x = nn.functional.relu(x + shortcut)

        x = self.final_conv(x)

        return x


class UNet(nn.Module):
    """U-Net with skip connections: two max-pool steps down, three
    transposed-convolution steps up, plus a residual block at the bottom.

    Doubles the lattice size, (N, 1, L, L) -> (N, 1, 2L, 2L); L must be
    divisible by 4. 8,402,369 parameters. This is the model used for all
    inverse RG results (thesis Listing A.7, Table A.1 "UNet for the inverse
    RG").
    """

    def __init__(self):
        super().__init__()

        self.Tconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),  # batch normalization
            #nn.Dropout2d(0.2),
            nn.ReLU(inplace = True)
        )

        self.Tconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),  # batch normalization
            #nn.Dropout2d(0.2),
            nn.ReLU(inplace = True)
        )

        self.Tconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),  # batch normalization
            #nn.Dropout2d(0.2),
            nn.ReLU(inplace = True)
        )

        self.pooling = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace = True)
        )

        self.pooling2 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace = True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(128),  # batch normalization
            nn.ReLU(inplace = True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # batch normalization
            nn.ReLU(inplace = True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(512),  # batch normalization
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace = True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(512),  # batch normalization
            nn.ReLU(inplace = True)  #added ReLu
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(512),  # batch normalization
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # batch normalization
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace = True)  #added ReLu
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(128),  # batch normalization
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace = True)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(64),  # batch normalization
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace = True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1,padding=0,padding_mode="circular")
        )

    def forward(self,x):

        x = self.conv1(x)

        shortcut1 = x

        x = self.pooling(x)

        x = self.conv2(x)

        shortcut2 = x

        x = self.pooling2(x)

        x = self.conv3(x)

        shortcut3 = x

        x = self.conv4(x)
        x = self.conv5(x)

        x = nn.functional.relu( x + shortcut3)

        x = self.Tconv1(x)

        x = torch.cat((x, shortcut2), dim=1)

        x = self.conv6(x)

        x = self.Tconv2(x)

        x = torch.cat((x, shortcut1), dim=1)

        x = self.conv7(x)

        x = self.Tconv3(x)

        x = self.conv8(x)

        x = self.final_conv(x)

        return x
