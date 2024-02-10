import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        conv_dropout: float = 0.0,
    ):
        super(DoubleConvolution, self).__init__()

        self.double_conv = nn.Sequential(
            # first conv
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=conv_dropout),
            # second conv
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=conv_dropout),
        )

    def forward(self, x):
        x = self.double_conv(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_dropout=0.0):
        super(Encoder, self).__init__()

        self.double_conv = DoubleConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_dropout=conv_dropout,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.double_conv(inputs)
        p = self.pool(x)

        return x, p


class Decoder(nn.Module):
    def __init__(self, out_channels: int, in_channels: int):
        super(Decoder, self).__init__()

        self.transpose = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2
        )

        """
        dubbio qui nel duble conv
        """
        self.double_conv = DoubleConvolution(
            in_channels=out_channels + out_channels,
            out_channels=out_channels,
        )

    def forward(self, inputs, skip):
        x = self.transpose(inputs)

        if x.shape != skip.shape:
            x = TF.resize(x, size=skip.shape[2:])

        x = torch.cat([x, skip], axis=1)
        x = self.double_conv(x)

        return x


class OUNET(nn.Module):
    def __init__(self, conv_dropout=0.0, final_dropout=0.0):
        super(OUNET, self).__init__()

        self.conv_dropout = conv_dropout
        self.final_dropout = final_dropout
        """
        Encoder part
        """
        self.encoder1 = Encoder(in_channels=1, out_channels=64)
        self.encoder2 = Encoder(in_channels=64, out_channels=128)
        self.encoder3 = Encoder(in_channels=128, out_channels=256)
        self.encoder4 = Encoder(in_channels=256, out_channels=512)

        """
        Bottom part
        """
        self.bottom = DoubleConvolution(512, 1024)

        """
        Decoder part
        """
        self.decoder1 = Decoder(in_channels=1024, out_channels=512)
        self.decoder2 = Decoder(in_channels=512, out_channels=256)
        self.decoder3 = Decoder(in_channels=256, out_channels=128)
        self.decoder4 = Decoder(in_channels=128, out_channels=64)
        """
        Final conv
        """
        self.convolution = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1  # binary segmentation
        )

    def forward(self, inputs):
        """
        Encoder
        """
        double_conv1, p1 = self.encoder1(inputs)
        double_conv2, p2 = self.encoder2(p1)
        double_conv3, p3 = self.encoder3(p2)
        double_conv4, p4 = self.encoder4(p3)

        """
        Bottom
        """
        bottom = self.bottom(p4)

        """
        Decoder
        """
        decoder1 = self.decoder1(bottom, double_conv4)
        decoder2 = self.decoder2(decoder1, double_conv3)
        decoder3 = self.decoder3(decoder2, double_conv2)
        decoder4 = self.decoder4(decoder3, double_conv1)

        """
        Final conv
        """
        output = self.convolution(decoder4)

        final_drop = nn.Dropout2d(p=self.final_dropout)
        output_drop = final_drop(output)

        return output_drop


if __name__ == "__main__":
    x = torch.randn((1, 512, 512))
    x = x.unsqueeze(0)

    f = OUNET()
    y = f(x)

    """
    La x ha 1, 3, 64, 64 featuers
    il 3 rappresenta il fatto che non riesco ad avere immagini in greyscale 
    Sono immagini RGB
    """
    print(x.shape)

    """
    La y ritorna 1, 1, 64, 64
    perche ritorna la maschera dell'immagine in binaco e nero
    """
    print(y.shape)
