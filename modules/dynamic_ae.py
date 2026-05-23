import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SpatialAttention, MultiHeadAttention
from .tools import ModuleTools, Reparameterizer
from .discriminator import PatchGAN

from bitnet import BitLinear
from vector_quantize_pytorch import ResidualVQ


def mmd_vae_loss(recon_x, x, z, mu, logvar, sigma=5.0):
    recon_loss = F.mse_loss(recon_x.clamp(0.0, 1.0), x.clamp(0.0, 1.0), reduction='mean') #/ x.size(0)
    # Calculate KL Loss
    kl = (1 + logvar - mu ** 2 - torch.exp(logvar))
    kl_per_img = -0.5 * torch.sum(kl, dim=-1)
    kl_loss = torch.mean(kl_per_img)

    # Sample prior
    prior_z = torch.randn_like(z)

    # Compute pairwise distances
    def compute_kernel(x, y):
        x_size, y_size = x.size(0), y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_diff = (x - y).pow(2).mean(2) / (2 * sigma ** 2)
        return torch.exp(-tiled_diff + 1e-8)

    # MMD loss
    xx = compute_kernel(z, z)
    yy = compute_kernel(prior_z, prior_z)
    xy = compute_kernel(z, prior_z)
    mmd_loss = xx.mean() + yy.mean() - 2 * xy.mean()

    return recon_loss, mmd_loss, kl_loss


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x) ** 2


class Downscale2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, use_attn=True, num_heads=4, use_bitnet=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        if use_attn:
            self.attn = SpatialAttention(out_channels, num_heads, use_bitnet=use_bitnet)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)

        if hasattr(self, "attn"):
            x = self.attn(x)

        return x 
    

class Upscale2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, out_padding=1, use_batchnorm=False, 
                 unet_style=False, unet_dropout=0.0, bias=False, use_attn=True, num_heads=4, use_bitnet=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=out_padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        if use_attn:
            self.attn = SpatialAttention(dim=out_channels, heads=num_heads, use_bitnet=use_bitnet)
        
        # self.unet_style = unet_style
        if unet_style:
            self.leaky_relu = nn.LeakyReLU(0.01)
            self.skip_dropout = nn.Dropout2d(unet_dropout)
            self.unet = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1, bias=bias),
                nn.BatchNorm2d(out_channels)
            )

    
    def forward(self, x, skip=None):
        if isinstance(x, tuple):
            x, skip = x
        x = self.conv_transpose(x)

        x = self.batch_norm(x) if self.batch_norm is not None else x
        if hasattr(self, "attn"):
            x = self.attn(x)

        if hasattr(self, "unet") and skip is not None:
            x = self.leaky_relu(x)

            skip = torch.zeros_like(x) if skip is None else self.skip_dropout(skip)

            x = torch.cat([x, skip], dim=1)
            x = self.unet(x)
            # x = self.conv(x)
            # x = self.batch_norm_conv(x)
        
        return x
        
class DynamicEncoder2D(nn.Module, ModuleTools):
    def __init__(self, latent_dims, channels=3, hidden_size=6256, use_attn=True, unet_style=False, num_heads=4, conv_bottleneck=None, use_bitnet=True):
        super().__init__()
        self.config = {
            "latent_dims": latent_dims,
            "channels": channels,
            "hidden_size": hidden_size,
            "use_attn": use_attn,
            "unet_style": unet_style,
            "conv_bottleneck": conv_bottleneck,
            "use_bitnet": use_bitnet
        }

        self.down_1 = nn.Sequential(Downscale2D(channels, 16,use_attn=False), nn.LeakyReLU(0.01, True))
        self.down_2 = nn.Sequential(Downscale2D(16, 32, use_attn=False), nn.LeakyReLU(0.01, True))
        self.down_3 = nn.Sequential(Downscale2D(32, 64, use_attn=False), nn.LeakyReLU(0.01, True))
        self.down_4 = nn.Sequential(Downscale2D(64, 128, use_attn=False), nn.LeakyReLU(0.01, True))
        self.down_5 = nn.Sequential(Downscale2D(128, 256, use_attn=False), nn.LeakyReLU(0.01, True))
        self.down_6 = Downscale2D(256, 512, use_attn=use_attn, num_heads=num_heads, use_bitnet=use_bitnet)

        self.unet_style=unet_style

        if conv_bottleneck:
            self.attn = SpatialAttention(512, num_heads, use_bitnet=False)

            self.mu = nn.Conv2d(512, conv_bottleneck, 3)
            self.logvar = nn.Conv2d(512, conv_bottleneck, 3)

        else:
            self.flatten = nn.Flatten(1)
            self.mu = nn.Linear(hidden_size, latent_dims) if not use_attn else MultiHeadAttention(hidden_size, latent_dims, num_heads) #if not use_bitnet else BitLinear(hidden_size, latent_dims)#MultiHeadAttention(hidden_size, latent_dims, num_heads, use_bitnet)
            self.logvar = nn.Linear(hidden_size, latent_dims) if not use_attn else MultiHeadAttention(hidden_size, latent_dims, num_heads) #if not use_bitnet else BitLinear(hidden_size, latent_dims)#MultiHeadAttention(hidden_size, latent_dims, num_heads, use_bitnet) #nn.Linear(hidden_size, latent_dims) if not use_bitnet else BitLinear(hidden_size, latent_dims)
        self.down_shape = ()
    
    def forward(self, x, return_skips=True):
        if self.unet_style and return_skips:
            skips = []
        x = self.down_1(x)
        
        if self.unet_style and return_skips:
            skips.append(x)

        x = self.down_2(x)
        
        if self.unet_style and return_skips:
            skips.append(x)

        x = self.down_3(x)
        
        if self.unet_style and return_skips:
            skips.append(x)

        x = self.down_4(x)

        if self.unet_style and return_skips:
            skips.append(x)

        x = self.down_5(x)

        if self.unet_style and return_skips:
            skips.append(x)

        x = self.down_6(x)

        if hasattr(self, "attn"):
            x = self.attn(x)

        if self.down_shape != x.shape:
            self.down_shape = x.shape

        if hasattr(self, "flatten"):
            x = self.flatten(x)
        
        try:
            if self.unet_style and return_skips:
                return self.mu(x), self.logvar(x), skips
                
            return self.mu(x), self.logvar(x)
        
        except Exception as ex:
            print("Try hidden_size:", x.shape[-1])
            raise ex


class DynamicDecoder2D(nn.Module, ModuleTools):
    def __init__(self, latent_dims, channels=3, hidden_size=6256, unflatten_shape=(128, 391), use_attn=True, unet_style=False, skip_dropout=0.2, num_heads=4, conv_bottleneck=None,
                 use_bitnet=True):
        super().__init__()

        self.config = {
            "latent_dims": latent_dims,
            "channels": channels,
            "hidden_size": hidden_size,
            "unflatten_shape": unflatten_shape,
            "use_attn": use_attn,
            "unet_style": unet_style,
            "skip_dropout": skip_dropout,
            "conv_bottleneck": conv_bottleneck,
            "use_bitnet": use_bitnet,
            "num_heads": num_heads
        }
        self.up_1 = nn.Sequential(Upscale2D(512, 256, use_batchnorm=True, unet_style=unet_style, unet_dropout=skip_dropout, use_attn=use_attn, num_heads=num_heads, use_bitnet=use_bitnet, out_padding=0), nn.LeakyReLU(0.01))
        self.up_2 = nn.Sequential(Upscale2D(256, 128, use_batchnorm=True, unet_style=unet_style, unet_dropout=skip_dropout, use_attn=False), nn.LeakyReLU(0.01))
        self.up_3 = nn.Sequential(Upscale2D(128, 64, use_batchnorm=True, unet_style=unet_style, unet_dropout=skip_dropout, use_attn=False), nn.LeakyReLU(0.01))
        self.up_4 = nn.Sequential(Upscale2D(64, 32, use_batchnorm=True, unet_style=unet_style, unet_dropout=skip_dropout, use_attn=False), nn.LeakyReLU(0.01))
        self.up_5 = nn.Sequential(Upscale2D(32, 16, use_batchnorm=True, unet_style=unet_style, unet_dropout=skip_dropout, use_attn=False), nn.LeakyReLU(0.01))
        self.up_6 = nn.Sequential(Upscale2D(16, channels, use_attn=False))

        if conv_bottleneck:
            self.bottleneck = nn.ConvTranspose2d(conv_bottleneck, 512, 3)#, SpatialAttention(128, num_heads))
            self.attn = SpatialAttention(512, num_heads, use_bitnet=False)
            
        else:
            self.bottleneck = nn.Sequential(nn.Unflatten(1, unflatten_shape)
                                            ,nn.Linear(latent_dims, hidden_size) if not use_attn else MultiHeadAttention(latent_dims, hidden_size, num_heads),
                                            nn.LeakyReLU(0.01))#nn.Sequential(MultiHeadAttention(latent_dims, hidden_size, num_heads, use_bitnet), nn.LeakyReLU(0.01) if not use_bitnet else SquaredReLU())

    def forward(self, x, skips=None):
        # x = self.fc(x)
        # x = self.unflatten(x)
        x = self.bottleneck(x)

        if hasattr(self, "attn"):
            x = self.attn(x)

        if skips is not None:
            skips = list(reversed(skips))

        x = self.up_1((x, None if skips is None else skips[0]))
        x = self.up_2((x, None if skips is None else skips[1]))
        x = self.up_3((x, None if skips is None else skips[2]))
        x = self.up_4((x, None if skips is None else skips[3]))
        x = self.up_5((x, None if skips is None else skips[4]))
        x = self.up_6(x)

        return x

    # def freeze_unet_layers(self):
        

class DynamicAutoencoder2D(nn.Module, ModuleTools, Reparameterizer):
    def __init__(self, latent_dims, channels, hidden_size=6256, unflatten_shape=(16, 391), num_quantizers=8, codebook_size=512, 
                 use_attn=True, use_bitnet=True, unet_style=False, skip_dropout=0.2, num_heads=4, conv_bottleneck=None):
        super().__init__()

        self.config = {
            "latent_dims": latent_dims,
            "channels": channels,
            "hidden_size": hidden_size,
            "use_attn": use_attn,
            "use_bitnet": use_bitnet,
            "unflatten_shape": unflatten_shape,
            "num_quantizers": num_quantizers,
            "unet_style": unet_style,
            "skip_dropout": skip_dropout,
            "num_heads": num_heads,
            "codebook_size": codebook_size,
            "conv_bottleneck": conv_bottleneck
        }

        self.encoder = DynamicEncoder2D(latent_dims, channels, hidden_size, use_attn, unet_style, num_heads, conv_bottleneck, use_bitnet)
        self.decoder = DynamicDecoder2D(latent_dims, channels, hidden_size, unflatten_shape, use_attn, unet_style, skip_dropout, num_heads, conv_bottleneck, use_bitnet)

        if not conv_bottleneck:
            self.rvq = ResidualVQ(dim = latent_dims, num_quantizers=num_quantizers, codebook_size=codebook_size)

            self.noise_pred = nn.Sequential(
                nn.Linear(latent_dims, latent_dims * 2) if not use_bitnet else BitLinear(latent_dims, latent_dims * 2),
                nn.LeakyReLU(0.01) if not use_bitnet else SquaredReLU(),
                nn.Linear(latent_dims * 2, latent_dims) if not use_bitnet else BitLinear(latent_dims * 2, latent_dims),
                nn.LeakyReLU(0.01) if not use_bitnet else SquaredReLU(),
                nn.Linear(latent_dims, 1) if not use_bitnet else BitLinear(latent_dims, 1))
        else:
            self.noise_pred = PatchGAN(16, 32, padding=2)

        self.unet_style = unet_style
    
    def to(self, *args, **kwargs):
        self.device = kwargs.get("device")
        
        return super().to(*args, **kwargs)

    def forward(self, x, y=None, t=None, sigma=5.0, use_skips=False):
        if self.unet_style and use_skips:
            mu, logvar, skips = self.encoder(x, use_skips)
        
        else:
            mu, logvar = self.encoder(x, False)
            skips = None

        z = self.reparameterize(mu, logvar)

        if hasattr(self, "rvq"):
            z, indices, commit_loss = self.rvq(z)
        else:
            indices, commit_loss = None, None

        try:
            recon = self.decoder(z, skips)

        except Exception as ex:
            print("Try Unflatten Shape:", self.encoder.down_shape)
            raise ex
        if t is not None:
            noise_pred = self.noise_pred(z)

        if y is not None:
            recon_loss, mmd_loss, kl_loss = mmd_vae_loss(recon, y, z, mu, logvar, sigma)#mmd_vae_loss(recon, y, z, sigma)
            # stft_loss = multi_scale_stft_loss(recon, y)
            if commit_loss is not None:
                losses = {
                    "recon_loss": recon_loss,
                    "kl_loss": kl_loss,
                    "commit_loss": commit_loss.mean(),
                    "mmd_loss": mmd_loss
                    # "stft_loss": stft_loss
                }
            else:
                losses = {
                    "recon_loss": recon_loss,
                    "kl_loss": kl_loss,
                    "mmd_loss": mmd_loss
                }
            if t is not None:
                print(noise_pred.shape)
                t_expanded = t.view(t.shape[0], *([1] * (noise_pred.ndim - 1))).expand_as(noise_pred)
                print(t_expanded.shape)

                noise_loss = F.mse_loss(noise_pred, torch.ones_like(noise_pred) * t_expanded)
                losses["noise_loss"] = noise_loss
            
            return z, recon, indices, losses

        return z, recon, indices, None

    def freeze_all_unet_layers(self):
        frozen = 0
        for module in self.modules():           # This walks through every submodule
            if isinstance(module, Upscale2D) and hasattr(module, "unet"):
                for param in module.unet.parameters():
                    param.requires_grad = False
                    frozen += 1
                # Optional: put in eval mode too
                # module.unet.eval()
        print(f"Froze {frozen} layers.")

if __name__ == "__main__":
    rand = torch.randn((2, 3, 128, 128))


    model = DynamicAutoencoder2D(128, 3, conv_bottleneck=8)
    model.print_parameters()
    z, recon, _, _ = model(rand)



    print("Latent Shape:", z.shape)

    print("Recon Shape:", recon.shape)