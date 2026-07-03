import torch
import torch.nn as nn

from sc_utils.nn import ModuleTools
    
class PatchGAN(nn.Module, ModuleTools):
    def __init__(self, input_channels, start_dim=64, depth=3, kernel_size=4, padding=1, leaky_relu_slope=0.2):
        super().__init__()

        self.config = {
            "input_channels": input_channels,
            "start_dim": start_dim,
            "depth": depth,
            "kernel_size": kernel_size,
            "padding": padding,
            "leaky_relu_slope": leaky_relu_slope
        }
        current_filters = start_dim
        layers = nn.ModuleList()

        layers.append(nn.Conv2d(input_channels, current_filters, kernel_size=3, stride=2, padding=padding))
        layers.append(nn.LeakyReLU(leaky_relu_slope))

        for i in range(depth):
            stride = 2 if i != depth - 1 else 1

            out_channels = current_filters * 2

            layers.append(nn.Conv2d(current_filters, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))

            current_filters = out_channels
        
        layers.append(nn.Conv2d(current_filters, 1, kernel_size=kernel_size, stride=1, padding=padding))

        for layer in layers:
            init_weights(layer)

        self.model = nn.Sequential(*layers)

    

    def forward(self, input):
        return self.model(input)
    

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight.data, 0.0, 0.2)
    
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


if __name__ == "__main__":
    rand = torch.randn(2, 3, 224, 224) # -> [2 x 1]

    pg = PatchGAN(3)

    # print(pg)

    output = pg(rand)
    print(output)
    print(output.shape)