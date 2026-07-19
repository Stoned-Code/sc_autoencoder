import gc
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
# from sc_utils.nn import ModuleTools, Reparameterizer

try:
    from .discriminator import PatchGAN
    from .attention import SpatialAttention
    from .tools import ModuleTools, Reparameterizer
except:
    try:
        from modules.discriminator import PatchGAN
        from modules.attention import SpatialAttention
        from modules.tools import ModuleTools, Reparameterizer
    except:
        from discriminator import PatchGAN
        from attention import SpatialAttention
        from tools import ModuleTools, Reparameterizer
def mmd_vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x.clamp(0.0, 1.0), x.clamp(0.0, 1.0), reduction='mean') #/ x.size(0)
    # Calculate KL Loss
    kl = (1 + logvar - mu ** 2 - torch.exp(logvar))
    kl_per_img = -0.5 * torch.sum(kl, dim=-1)
    kl_loss = torch.mean(kl_per_img)


    return recon_loss, kl_loss

class Downscale2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, slope=0.01):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(slope)
    
    def forward(self, data):
        x, skips = data[0], data[1]
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        if skips is not None:
            skips.append(x)
        
        return x, skips

class COEncoder(nn.Module, ModuleTools):
    def __init__(self, color_channels=2, start_channels=16, depth=6, bottleneck=8, leaky_relu_slope=0.01, num_heads=4, use_bitnet=True, stride=[], padding=[]):
        super().__init__()
        self.config = {
            "color_channels": color_channels,
            "start_channels": start_channels,
            "depth": depth,
            "bottleneck": bottleneck,
            "leaky_relu_slope": leaky_relu_slope,
            "num_heads": num_heads,
            "use_bitnet": use_bitnet,
            "stride": stride.copy(),
            "padding": padding.copy()
        }

        self.convs = [Downscale2d(color_channels, start_channels, 3, 2 if len(stride) == 0 else stride.pop(0), 1 if len(padding) == 0 else padding.pop(0), False)]

        # self.convs = nn.ModuleList([nn.Sequential(nn.Conv2d(color_channels, start_channels, 3, 2 if len(stride) == 0 else stride.pop(0), 1 if len(padding) == 0 else padding.pop(0), bias=False),
        #                                           nn.BatchNorm2d(start_channels),
        #                                           nn.LeakyReLU(leaky_relu_slope))])

        self.max_channels = start_channels

        for i in range(depth - 1):
            self.convs.append(Downscale2d(self.max_channels, self.max_channels * 2, 3, 2 if len(stride) == 0 else stride.pop(0), 1 if len(padding) == 0 else padding.pop(0), False))
            # self.convs.append(nn.Sequential(nn.Conv2d(self.max_channels, self.max_channels * 2, 3, 2 if len(stride) == 0 else stride.pop(0), 1 if len(padding) == 0 else padding.pop(0), bias=False),
            #                                 nn.BatchNorm2d(self.max_channels * 2),
            #                                 nn.LeakyReLU(leaky_relu_slope)))

            self.max_channels = self.max_channels * 2
        
        self.convs = nn.Sequential(*self.convs)

        self.spatial_attention = SpatialAttention(self.max_channels, num_heads, use_bitnet)
        
        self.mu = nn.Conv2d(self.max_channels, bottleneck, 3)
        self.logvar = nn.Conv2d(self.max_channels, bottleneck, 3)
    
    def forward(self, x, return_skips=False):

        if return_skips:
            skips = []
        else:
            skips = None
        # logits = x
        # for i, conv in enumerate(self.convs):
        #     logits = conv(logits)

        #     if return_skips and i < len(self.convs) - 1:
        #         skips.append(logits)

        logits, skips = self.convs((x, skips))

        # if skips is not None:
        #     skips = skips[-4:]

        logits = self.spatial_attention(logits)
        # print(logits.shape)
        if skips is not None:
            _ = skips.pop()
        return self.mu(logits), self.logvar(logits), skips

class Upscale2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, slope = 0.01, use_unet=False, use_activation=True):
        super().__init__()
        if use_activation:
            self.transpose = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(slope))
        else:
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

        if use_unet:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels))

    def forward(self, data):
        x, skips = data[0], data[1]
        x = self.transpose(x)
        # x = self.batch_norm(x)
        # x = self.relu(x)

        if hasattr(self, "conv") and skips is not None and len(skips) > 0:
            skip = skips.pop()
            _, _, l_h, l_w = x.shape
            _, _, s_h, s_w = skip.shape

            if x.shape != skip.shape:
                pad_h, pad_w = l_h - s_h , l_w - s_w
                # skip = skips.pop()
                skip = F.pad(skip, (0, pad_h, 0, pad_w))
            
            # print(x.shape)
            # print(skip.shape)
            # print(i)    
            skip = torch.cat([x, skip], dim=1)
            x = self.conv(skip)
        return x, skips
    

class CODecoder(nn.Module, ModuleTools):
    def __init__(self, color_channels=2, start_channels=512, depth=6, bottleneck=8, out_padding=[],
                 leaky_relu_slope=0.01, use_unet=False, num_heads=4, use_bitnet=True, stride=[]):
        super().__init__()
        self.config = {
            "color_channels": color_channels,
            "start_channels": start_channels,
            "depth": depth,
            "bottleneck": bottleneck,
            "out_padding": out_padding.copy(),
            "leaky_relu_slope": leaky_relu_slope,
            "use_unet": use_unet,
            "num_heads": num_heads,
            "use_bitnet": use_bitnet,
            "stride": stride.copy()
        }
        self.bottleneck = nn.ConvTranspose2d(bottleneck, start_channels, 3)
        self.transposes = nn.ModuleList()
        self.spacial_attention = SpatialAttention(start_channels, num_heads, use_bitnet)
        # if use_unet:
        #     self.convs = []
            # conv_count = 0
        
        self.min_channels = start_channels

        for i in range(depth - 1):
            self.transposes.append(Upscale2d(self.min_channels, self.min_channels // 2, 3, 2 if len(stride) == 0 else stride.pop(0), 1, 1 if len(out_padding) == 0 else out_padding.pop(0), leaky_relu_slope, use_unet, True))

            self.min_channels = self.min_channels // 2

        self.transposes.append(Upscale2d(self.min_channels, color_channels, 3, 2 if len(stride) == 0 else stride.pop(0), 1, 1 if len(out_padding) == 0 else out_padding.pop(0), use_activation=False))      
        self.transposes = nn.Sequential(*self.transposes)
        # self.transposes.append(nn.ConvTranspose2d(self.min_channels, color_channels, 3, 2 if len(stride) == 0 else stride.pop(0), 1, 1 if len(out_padding) == 0 else out_padding.pop(0)))

    @property
    def depth(self):
        return self.config["depth"]

    def forward(self, x, skips: list=None):
        logits = self.bottleneck(x)
        logits = self.spacial_attention(logits)
        # if hasattr(self, "convs"):
        #     conv_idx = 0
        logits, _ = self.transposes((logits, skips))

        # for i, transpose in enumerate(self.transposes):
        #     logits = transpose(logits)

        #     if hasattr(self, "convs") and skips is not None and len(skips) > 0:# and i < 4:
        #         skip = skips.pop()
        #         _, _, l_h, l_w = logits.shape
        #         _, _, s_h, s_w = skip.shape

        #         if logits.shape != skip.shape:
        #             pad_h, pad_w = l_h - s_h , l_w - s_w
        #             # skip = skips.pop()
        #             skip = F.pad(skip, (0, pad_h, 0, pad_w))
                
        #         # print(logits.shape)
        #         # print(skip.shape)
        #         # print(i)    
        #         skip = torch.cat([logits, skip], dim=1)
        #         logits = self.convs[i](skip)
        #         # conv_idx += 1

        return logits

class ImageAE(nn.Module, ModuleTools, Reparameterizer):
    def __init__(self, color_channels=3, start_channels=16, depth=4, bottleneck=8, leaky_relu_slope=0.01, output_padding=[],
                 unet_style=False, num_heads=4, use_bitnet=True):
        super().__init__()

        self.config = {
            "color_channels": color_channels,
            "start_channels": start_channels,
            "depth": depth,
            "bottleneck": bottleneck,
            "leaky_relu_slope": leaky_relu_slope,
            "output_padding": output_padding.copy(),
            "unet_style": unet_style,
            "num_heads": num_heads,
            "use_bitnet": use_bitnet
        }

        

        stride = list(reversed([p + 1 for p in output_padding]))
        self.enc = COEncoder(color_channels, start_channels, depth, bottleneck, leaky_relu_slope, num_heads, use_bitnet, stride.copy(), list(reversed(output_padding.copy())))
        self.dec = CODecoder(color_channels, self.enc.max_channels, depth, bottleneck, output_padding, leaky_relu_slope, unet_style, num_heads, use_bitnet, list(reversed(stride.copy())))
        self.noise_predictor = PatchGAN(bottleneck, start_channels, 2, 4)

    @property
    def unet_style(self):
        return self.config["unet_style"]

    def forward(self, x, y=None, t=None, use_skips=False):
        z, mu, logvar, skips = self.encode(x, use_skips)

        recon = self.dec(z, skips)

        if t is not None:
            noise_pred = self.noise_predictor(z)
            
        if y is not None:
            recon_loss, kl_loss = mmd_vae_loss(recon, y, mu, logvar)

            losses = {
                "recon_loss": recon_loss,
                "kl_loss": kl_loss
            }

            if t is not None:
                stack = []

                for tb, np in zip(t, noise_pred):
                    stack.append(torch.full_like(np, tb))
                
                noise_loss = F.mse_loss(noise_pred, torch.stack(stack))
                losses["noise_loss"] = noise_loss

            return z, recon, losses
        
        return z, recon, None

    def encode(self, x, use_skips):
        mu, logvar, skips = self.enc(x, use_skips)
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar, skips

    def decode(self, z):
        return self.dec(z)
    
    def freeze_all_unet_layers(self):
        frozen = 0
        for module in self.modules():           # This walks through every submodule
            if isinstance(module, Upscale2d) and hasattr(module, "conv"):
                for param in module.conv.parameters():
                    param.requires_grad = False
                    frozen += 1
                # Optional: put in eval mode too
                # module.unet.eval()
        print(f"Froze {frozen} layers.")

def get_padding(required_shape, depth, bottleneck, start_channels, stride, padding, allowed_failur=20):
    attempts = []
    x = torch.randn(required_shape)
    attempt = [random.choice([0, 1]) for _ in range(depth)]
    attempts.append(attempt)
    model = ImageAE(required_shape[1], start_channels, depth, bottleneck, 0.01, attempt.copy(), False, stride=stride,padding=padding)
    
    _, recon, _ = model(x)
    attempt_count = 1
    while recon.shape != required_shape:
        for i in range(allowed_failur):
            attempt = [random.choice([0, 1]) for _ in range(depth)]
            if attempt not in attempts:
                break
        else:
            print("Failed to find solution...")
            return []
        
        attempts.append(attempt)
        model = ImageAE(required_shape[1], start_channels, depth, bottleneck, 0.01, attempt.copy(), False, stride=stride,padding=padding)
        _, recon, _ = model(x)
        attempt_count += 1
        print(f"Attempt: {attempt_count}", end="\r")
    print("Successful:", attempt)

    return attempt


# if __name__ == "__main__":
#     scale = 512
#     x = torch.randn((2, 3, 512, 224))
#     start_channels = 8
#     depth = 6
#     bottleneck = 16

#     paddings = [1, 1, 1, 1, 0, 0]

#     model = ImageAE(3, start_channels, depth, bottleneck, 0.01, paddings, True)

#     z, recon, _ = model(x, use_skips=True)

#     print("Input:", x.shape)
#     print("Latent:", z.shape)
#     print("Recon:", recon.shape)

#     z = torch.randn((2, 16, 2, 2))

#     recon = model.decode(z)
#     print(recon.shape)

if __name__ == "__main__":
    import lpips
    loss_fn_alex = lpips.LPIPS(net="alex")
    img_0 = torch.randn((3, 3, 224, 224))
    img_1 = torch.randn((3, 3, 224, 224))
    img_0 = (img_0 - 0.5) * 2
    img_1 = (img_1 - 0.5) * 2
    loss = loss_fn_alex(img_0, img_1).mean()
    print(loss.shape)
    print(loss)
    print(img_0.shape)
    resized = F.interpolate(img_0, size=(64, 64), mode="bilinear", align_corners=False)
    # resized = val.resize(3, 3, 64, 64)
    print(resized.shape)