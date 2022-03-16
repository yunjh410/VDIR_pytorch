import torch
from torch import nn
from torch.nn import Parameter
import math
import numpy as np
import logging


def conv2d(in_, out_, stride=1):
    return torch.nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=True, padding_mode='zeros')
    
def save_checkpoint(path, model, discriminator, optimizer, optimizer_d, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict' : discriminator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict()     
    }, path)
    
def psnr(img1, img2):
    img1=np.float32(img1)
    img2=np.float32(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX= 1.0
    else:
        PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Spectral Normalization from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    
    
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def logger(log_adr, logger_name, mode='w'):
    """
    Logger
    """
    # create logger
    _logger = logging.getLogger(logger_name)
    # set level
    _logger.setLevel(logging.INFO)
    # set format
    formatter = logging.Formatter('%(message)s')
    # stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    # file
    file_handler = logging.FileHandler(log_adr, mode=mode)
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    return _logger

def date_time(secs):
    day = secs // (24 * 3600)
    secs = secs % (24 * 3600)
    hour = secs // 3600
    secs %= 3600
    minutes = secs // 60
    secs %= 60
    seconds = int(secs)
    return f'{day} d {hour} h {minutes} m {seconds} s'