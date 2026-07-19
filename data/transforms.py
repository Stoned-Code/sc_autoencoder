import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TT
import torch
from PIL import Image, ImageOps
import numpy as np
from enum import Enum
from sc_utils.processing.image_processing import set_shortest_length
from io import BytesIO

def is_solid_color_tensor(img: torch.Tensor) -> bool:
    """Check if a torch tensor image (C,H,W) is a solid single color."""
    if not isinstance(img, torch.Tensor) or img.dim() != 3:
        return False
    
    c, h, w = img.shape
    if h == 0 or w == 0:
        return False
    
    # Compare everything against the first pixel
    first_pixel = img[:, 0, 0]          # shape (C,)
    
    return bool(torch.all(img == first_pixel.view(c, 1, 1)))


def get_timestep_embedding(t, dim, max_period=10000):
    """
    Standard sinusoidal timestep embedding used in diffusion models.
    """
    half = dim // 2
    freqs = torch.arange(half, dtype=torch.float32, device=t.device)
    freqs = max_period ** (-freqs / half)
    
    angles = t[:, None] * freqs[None, :]   # [B, half]
    
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, dim]
    
    # Optional: small linear projection if dim is large
    # if dim % 2 != 0:
    #     emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    
    return emb


class TileImageTensor:
    def __init__(self, tile_size, return_random_tile=True, random_attempts = 10):
        self.tile_size = tile_size
        self.return_random_tile = return_random_tile
        self.random_attempts = random_attempts

    def __call__(self, x):
        c, _, _ = x.shape

        x = x.unsqueeze(0).float()
        #print(x.shape)
        patches = F.unfold(x, kernel_size=self.tile_size, stride=self.tile_size)
        patches = patches.view(1, c, self.tile_size, self.tile_size, -1)
        patches = patches.permute(0, 4, 1, 2, 3).squeeze(0)
        if self.return_random_tile:
            ind = random.randint(0, patches.shape[0] - 1)
            patch = patches[ind]
            for _ in range(self.random_attempts):
                if not is_solid_color_tensor(patch):
                    break

                ind = random.randint(0, patches.shape[0] - 1)
                patch = patches[ind]

            
            return patch
        
        return patches

class AddTimestepNoise:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        # These become persistent attributes — safe even across worker processes
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def __call__(self, x, ts_sigma=1.0):
        # x is usually a PIL Image or already a torch.Tensor in [0,1]
        # Make sure it's a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 255.0
        
        # Sample timestep (1 to T)
        t = torch.randint(1, int(self.T * ts_sigma) + 1, (1,)).item()   # scalar is fine here
        
        # Grab precomputed values (CPU → CPU, very fast)
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t - 1]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t - 1]
        
        # Add noise — still scalar broadcasting
        noise = torch.randn_like(x)
        x_t = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t, t   # most people return both so the model knows which t to denoise from


class SquareMethod(Enum):
    CROP = 0
    PAD = 1
    TILE = 2


class PILSquareTransform:
    def __init__(self, window, method, tile_size, random_tile=True, grayscale=False):
        #super(PILSquareTransform, self).__init__()
        self.window = window
        self.method = method
        self.random_tile = random_tile
        self.grayscale = grayscale
        if method == SquareMethod.TILE:
            self.tiler = TileImageTensor(tile_size, random_tile)
            self.to_tensor = TT.PILToTensor()
            self.tile_size = tile_size

    def __call__(self, img: Image):
        img = img.convert("RGB" if not self.grayscale else "L")

        w, h = img.size

        if self.method == SquareMethod.CROP:
            if w != h:
                smallest = min(w, h)
                if self.window is None:
                    if self.random_tile:
                        offset = random.randint(0, abs(w - h))
                    else:
                        offset = abs(w - h) // 2
                else:
                    offset = self.window

                if w > h:
                    img = img.crop((offset, 0, offset + smallest, h))
                else:
                    img = img.crop((0, offset, w, offset + smallest))
        
        elif self.method == SquareMethod.PAD:
            img = ImageOps.pad(img, (max(img.size), max(img.size)), color=0)

        elif self.method == SquareMethod.TILE:
            if min(w, h) < self.tile_size:
                img = set_shortest_length(img, self.tile_size)

            img = self.to_tensor(img)
            img = self.tiler(img)
        return img


class ImageNormalize:
    def __init__(self, bidirection=False):
        self.bidirectional = bidirection

    def __call__(self, img):
        if not self.bidirectional:
            img = img / 255.0
        else:
            img = (img - 127.5) / 127.5

        return img


class SquarePad:#(nn.Module):
    def __init__(self):
        #super(SquarePad, self).__init__()
        pass
        
    def __call__(self, img: Image):
        C, H, W = img.shape
        if H != W:
            extra = max([H, W]) - min([H, W])
            side_1 = extra // 2
            side_2 = extra - side_1

            if H > W:
                side_1_t = torch.zeros((C, H, side_1), dtype=img.dtype)
                side_2_t = torch.zeros((C, H, side_2), dtype=img.dtype)

                img = torch.cat([side_1_t, img, side_2_t], dim=2)
            else:
                side_1_t = torch.zeros((C, side_1, W), dtype=img.dtype)
                side_2_t = torch.zeros((C, side_2, W), dtype=img.dtype)

                img = torch.cat([side_1_t, img, side_2_t], dim=1)

        return img

class Bytes2Image:
    def __init__(self, convert=None):
        self.convert = convert

    def __call__(self, b):
        img = Image.open(BytesIO(b))
        if self.convert is None:
            return img

        return img.convert(self.convert) 
    

class SquareCrop:#(nn.Module):
    def __init__(self, window_scalar = None):
        #super(SquareCrop, self).__init__()
        self.window = window_scalar

    def __call__(self, img: Image):
        try:
            C, H, W = img.shape
        except ValueError as ex:
            print("Image Shape:", img.shape)
            raise ex

        if H != W:
            smallest = min([H, W])
            window = max([H, W]) - min([H, W])
            if self.window is None:
                rand = int(window * random.random())
            else:
                rand = self.window

            if H > W:
                img = img[:, rand:rand+smallest, :]
            else:
                img = img[:, :, rand:rand+smallest]

        return img


class SquareImageTransform:
    def __init__(self,
                side_length=64,
                square_method = SquareMethod.CROP,
                bidirection=False,
                window=None,
                denoise=False,
                TS=1000,
                random_tile=True,
                grayscale=False):


        self.side_length = side_length
        self.reshape = PILSquareTransform(window, square_method, side_length, random_tile, grayscale)
        self.img_norm = ImageNormalize(bidirection)
        if square_method != SquareMethod.TILE:
            self.to_tensor = TT.PILToTensor()
        self.denoise = denoise
        self.add_noise = AddTimestepNoise(TS)
        self.square_method = square_method
        self.ts_sigma = 1.0

    def set_ts_sigma(self, ts_sigma):
        self.ts_sigma = ts_sigma
    
    def __call__(self, x):
        # if isinstance(x, list):
        #     return self.__transformitems__(x, self.ts_sigma)
        # elif isinstance(x, Image):
        # print(type(x))
        return self.__transform__(x[0], self.ts_sigma)

    def __transformitems__(self, imgs, ts_sigma):
        x_stack = []
        y_stack = []
        t_stack = [] if self.denoise else None

        for img in imgs:
            x, y, t = self.__transform__(img, ts_sigma)
            x_stack.append(x)
            y_stack.append(y)

            if t_stack is not None:
                t_stack.append(t)
        
        if t_stack is not None:
            # print(t_stack)
            t_stack = torch.stack(t_stack)

        return torch.stack(x_stack), torch.stack(y_stack), t_stack


    def __transform__(self, img: Image, ts_sigma):
        # index = self.indexes[idx]

        # img = self.dataset[index]
        if isinstance(img, bytes):
            img = Image.open(BytesIO(img))

        if self.square_method != SquareMethod.TILE:
            img = self.reshape(img)
            img = img.resize((self.side_length, self.side_length), Image.Resampling.LANCZOS)
        
            img = self.to_tensor(img)

        if self.square_method == SquareMethod.TILE:
            #img = self.to_tensor(img)
            img = self.reshape(img)

        #img = self.resize(img)


        img = self.img_norm(img)

        if self.denoise:
            noisy, t = self.add_noise(img, ts_sigma)
            # print("Max:", img.max())
            return noisy, img, torch.tensor(t)

        return img, img, 0
    

class TorchResize:
    def __init__(self, scale, mode="bilinear", align_corners=False):
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners
    
    def __call__(self, x):
        return F.interpolate(x, self.scale, mode=self.mode, align_corners=self.align_corners)
    