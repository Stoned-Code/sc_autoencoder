import lpips
import torch

class LPIPSLoss:
    NET_ALEX = "alex"
    NET_VGG = "vgg"
    def __init__(self, net=NET_ALEX, swap_norm=True):
        self.swap_norm = swap_norm
        self.fn = lpips.LPIPS(net=net)
        self.fn.eval()

        for p in self.fn.parameters(): p.requires_grad = False
    
    def to(self, device):
        self.fn = self.fn.to(device)
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        # print(x.device)
        # print(y.device)
        if self.swap_norm:
            x -= 0.5
            x *= 2

            y -= 0.5
            y *= 2

        loss = self.fn.forward(x,y)

        return loss.mean()