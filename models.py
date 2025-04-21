import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy

class TeLU(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'silu':
        return nn.SiLU(inplace=True)
    elif activation == 'mish':
        return nn.Mish(inplace=True)
    elif activation == 'telu':
        return TeLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x3 = self.conv7x7(x)
        return x1 + x2 + x3

class EdgeAwareConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeAwareConv, self).__init__()
        self.edge_filter = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        sobel_kernel = torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32  # Ensure float type
        )
        self.edge_filter.weight = nn.Parameter(
            sobel_kernel.repeat(in_channels, 1, 1, 1), requires_grad=True
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        edge_map = self.edge_filter(x)
        x = self.conv(x + edge_map)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LineDetectionModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, activation='relu'):
        super(LineDetectionModel, self).__init__()
        self.multi_scale = MultiScaleConv(in_channels, out_channels // 4)
        self.edge_aware = EdgeAwareConv(out_channels // 4, out_channels // 2)
        self.depthwise_separable = DepthwiseSeparableConv(out_channels // 2, out_channels)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.multi_scale(x)
        x = self.activation(self.edge_aware(x))
        x = self.activation(self.depthwise_separable(x))
        x = self.final_conv(x)
        return x

class ResNetBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, activation='relu'):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.act = get_activation(activation)
        self.shortcut = nn.Identity() if in_ch==out_ch else nn.Conv2d(in_ch,out_ch,1,bias=False)
    def forward(self, x):
        sc = self.shortcut(x)
        out = self.bn1(x); out = self.act(out)
        out = self.conv1(out)
        out = self.bn2(out); out = self.act(out)
        out = self.conv2(out)
        return out + sc

class NanoUpscaler(nn.Module):
    def __init__(self, in_channels=3, edge_feature_dim=64, resnet_feature_dim=64, resnet_layers=1, activation='relu'):
        super(NanoUpscaler, self).__init__()

        # === Identity Convolution for Residual Connection ===
        self.conv_identity = nn.Conv2d(in_channels, edge_feature_dim, kernel_size=1, stride=1, padding=0, bias=False)

        # === Line Detection ===
        self.line_detection = LineDetectionModel(in_channels=in_channels, out_channels=edge_feature_dim, activation=activation)

        # === Transpose Convolution ===
        self.conv_out = nn.ConvTranspose2d(edge_feature_dim, resnet_feature_dim, kernel_size=3, stride=1, padding=1, bias=False)

        # === Activation Function ===
        self.intermediate_activation = get_activation('relu')

        # === Residual ResNet Blocks ===
        resnet_blocks = []
        for _ in range(resnet_layers):
            resnet_blocks.append(ResNetBlock2(resnet_feature_dim, resnet_feature_dim, activation=activation)) # Input/output channels are now feature_dim
        self.resnets = nn.Sequential(*resnet_blocks)

        # === Identity for Residual Learning ===
        self.resnet_identity = nn.Conv2d(resnet_feature_dim, resnet_feature_dim, kernel_size=1) # Match feature dimensions

        # === Final Convolution to Output Channels ===
        self.conv_final = nn.Conv2d(resnet_feature_dim, in_channels, kernel_size=1)

        # === Final Activation ===
        self.final_activation = get_activation('relu')

    def forward(self, x):
        identity = self.conv_identity(x)
        x = self.line_detection(x)
        x = x + identity
        x = self.conv_out(x)
        x = self.intermediate_activation(x) # Activation after transpose

        # Residual Learning with ResNet
        resnet_input = self.resnet_identity(x) # Project to the same feature dimension
        residual = self.resnets(x)
        x = resnet_input + residual # Add the residual

        x = self.conv_final(x) # Project back to the output number of channels
        x = self.final_activation(x)
        return x

class PicoUpscaler(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_dim=64, activation='relu'):
        super(PicoUpscaler, self).__init__()      

        # === Identity Convolution for Residual Connection ===
        self.conv_identity = nn.Conv2d(in_channels, feature_dim, kernel_size=1, stride=1, padding=0, bias=False)

        # === Line Detection ===
        self.line_detection = LineDetectionModel(in_channels=in_channels, out_channels=feature_dim, activation=activation)

        # === Activation Function ===  
        self.final_activation = get_activation('relu')
        
        # === Transpose Convolution ===
        self.conv_out = nn.ConvTranspose2d(feature_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        identity = self.conv_identity(x)
        x = self.line_detection(x)        
        x = x + identity
        x = self.conv_out(x)
        x = self.final_activation(x)
        return x

    
def get_model(name = 'pico'):
    if name == 'pico':
        return PicoUpscaler(in_channels=3, out_channels=3, feature_dim=128, activation='relu')
    if name == 'nano':
        return NanoUpscaler(in_channels=3, edge_feature_dim=128, resnet_feature_dim=64, resnet_layers=1, activation='relu')

if __name__ == "__main__":

    x = torch.randn(1, 3, 752, 576).to("cuda")
    model = get_model('pico').to("cuda")
    model = torch.compile(model)

    # Warm-up
    for _ in range(10):
        _ = model(x)

    # Measure FPS over 20 seconds
    start_time = time.time()
    num_iterations = 0
    while time.time() - start_time < 20:
        _ = model(x)
        num_iterations += 1

    elapsed_time = time.time() - start_time
    fps = num_iterations / elapsed_time

    print("Model output shape:", model(x).shape)
    print("Model size: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model size (MB): ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    print(f"Average FPS: {fps:.2f}")
