# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
import cv2
import vision_transformer_mod as vits
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from pydensecrf.utils import unary_from_labels
from scipy.linalg import eigh
import timm
import torchvision.models as models
from models.vgg import VGG_Backbone

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def save_visualizations(num_images,masks_m,attn_maps,pseudo_masks,pts,count1):
    for i in range(num_images):
        pred = normalize_im(masks_m[i][0].detach().cpu().numpy())
        #print('pm:',pseudo_masks[i].detach().cpu().numpy().shape)

        dino = normalize_im(attn_maps[i].detach().cpu().numpy())
        pseudo = normalize_im(pseudo_masks[i].detach().cpu().numpy())
        org = cv2.imread('./datasets/COCO9213/img_bilinear_224/'+pts[i][0])
        org = cv2.resize(org,(224,224))

        dino = cv2.applyColorMap(dino, cv2.COLORMAP_JET)
        #dino = cv2.cvtColor(dino,cv2.COLOR_BGR2RGB)
        dino = cv2.addWeighted(dino, 0.5, org, 0.5, 0)

        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        #pred = cv2.cvtColor(pred,cv2.COLOR_BGR2RGB)
        pred = cv2.addWeighted(pred, 0.5, org, 0.5, 0)

        #pseudo = cv2.applyColorMap(pseudo, cv2.COLORMAP_JET)
        pseudo = cv2.cvtColor(pseudo,cv2.COLOR_GRAY2RGB)
        #pseudo = cv2.addWeighted(pseudo, 0.75, org, 0.25, 0)

#             enc_attn = normalize_im(enc_attn_maps2[i])
#             enc_attn = cv2.resize(enc_attn,(pseudo.shape[1],pseudo.shape[0]))
#             enc_attn = cv2.applyColorMap(enc_attn, cv2.COLORMAP_JET)
#             #pred = cv2.cvtColor(pred,cv2.COLOR_BGR2RGB)
#             enc_attn = cv2.addWeighted(enc_attn, 0.5, org, 0.5, 0)


        #dino = cv2.cvtColor(np.uint8(dino),cv2.COLOR_GRAY2RGB)
        #pred = cv2.cvtColor(np.uint8(pred),cv2.COLOR_GRAY2RGB)

        combo = np.concatenate((org,dino),0)
#             combo = np.concatenate((combo,enc_attn),0)
        combo = np.concatenate((combo,pseudo),0)
        combo = np.concatenate((combo,pred),0)

        if i == 0:
            combom = combo
        else:
            combom = np.concatenate((combom,combo),1)

    cv2.imwrite('./interim/im_'+str(count1)+'.png',combom)

        
def normalize_im(im):
    if np.max(im) > 0:
        im = (im - np.min(im))/(np.max(im)-np.min(im))
    im = np.uint8(im*255)
    return im

def normalize_im_torch(im):
    if torch.max(im) > 0:
        im = (im - torch.min(im))/(torch.max(im)-torch.min(im))
    return im

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = None
    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("We load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)["state_dict"]
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v

def SCloss(x, p, n):
    dist_p = (1 + compute_cos_dis(x, p)) * 0.5
    dist_n = (1 + compute_cos_dis(x, n)) * 0.5
    loss = - torch.log(dist_p + 1e-5) - torch.log(1. + 1e-5 - dist_n)
    return loss.sum()

def Embeddingloss(p, n):
    dist = (1 + compute_cos_dis(p, n)) * 0.5
    loss = - torch.log(dist + 1e-5)
    return loss.sum()

def Embedloss(protos):
    loss = 0
    cnt = 0
    for i in range(len(protos)):
        for j in range(i,len(protos)):
            #print('prot size:',protos[i,:,0].size())
            loss += Embeddingloss(protos[i,:,0],protos[j,:,0])
            cnt += 1
    return loss/cnt

def normalize_im(s_pred):
    if np.max(s_pred) != np.min(s_pred):
        s_pred = (s_pred - np.min(s_pred))/(np.max(s_pred)-np.min(s_pred))
        s_pred = np.uint8(s_pred*255)
    else:
        5==5
    return s_pred

def eval_pr(y_pred, y, num):
    prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
    thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall

def IoU_Loss(pred, target):
    b = pred.size()[0]
    IoU = 0.0
    for i in range(0, b):
        Iand1 = torch.sum(target[i, :, :, :]*pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :])-Iand1
        IoU1 = Iand1/(Ior1 + 1e-5)
        IoU = IoU + (1-IoU1)

    return IoU/b

def Eval_mae(preds,gts):
        avg_mae, img_num = 0.0, 0.0
        with torch.no_grad():
            list1 = []
            for i in range(len(preds)):
                pred = preds[i]
                gt = gts[i]
                pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
                
                mea = torch.abs(pred - gt).mean()
                if mea == mea:  # for Nan
                    list1.append(mea)
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num
            return avg_mae.item(),list1,avg_mae
            
def Eval_Emeasure(preds,gts):
    avg_em, img_num = 0.0, 0.0
    with torch.no_grad():
        list1 = []
        for i in range(len(preds)):
            pred = preds[i]
            gt = gts[i]
            pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
            Em = torch.zeros(255)
            pred = (pred - torch.min(pred)) / (torch.max(pred) -torch.min(pred) + 1e-20)
            Em += eval_e(pred, gt, 255)
            list1.append(Em)
            avg_em += Em
            img_num += 1.0

        Em = avg_em/img_num
        return Em.max().item(),list1,avg_em
        
def eval_e(y_pred, y, num):
        score = torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).float()
            fm = y_pred_th - y_pred_th.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
        return score
        
def Eval_fmeasure(preds,gts):
        beta2 = 0.3
        avg_f, img_num = 0.0, 0.0
        
        with torch.no_grad():
            list1 = []
            for i in range(len(preds)):
                pred = preds[i]
                gt = gts[i]
                pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
                prec, recall = eval_pr(pred, gt, 255)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0
                list1.append(f_score)
                avg_f += f_score
                img_num += 1.0
            Fm = avg_f / img_num
            return Fm.max().item(),list1,avg_f #torch.max(Fm)

def get_dino_feat(inp, backbone,patch_size, tau):
    tensor = inp
    feat = backbone(tensor)[0]
    return feat
        
def normalize_im(im):
    if np.max(im) > 0:
        im = (im-np.min(im))/(np.max(im)-np.min(im))
    else:
        im = im
    return im
    

def visualize_preds3(no_crf,cross_attwts,root1,root2,pths,epoch,idx): #edges,edges2,edges_pred, root2,

    for i in range(min(10,len(no_crf))): #len(preds)
        try:
            #print(root1+'/'+pths[i][0][:-4]+'.jpg')
            orig = cv2.imread(root1+'/'+pths[i][0][:-4]+'.jpg')
            orig = cv2.resize(orig,(224,224))
        except:
            #print(root1+'/'+pths[i][0][:-4]+'.png')
            orig = cv2.imread(root1+'/'+pths[i][0][:-4]+'.png')
            orig = cv2.resize(orig,(224,224))
        
        gt = cv2.imread(root2+'/'+pths[i][0][:-4]+'.png')
        
        pred3 = cross_attwts[i][0].detach().cpu().numpy()
        pred3 = cv2.resize(pred3,(224,224))
        pred3 = np.uint8(normalize_im(pred3)*255)
        vis_fix_map3 = cv2.resize(pred3,(orig.shape[1],orig.shape[0]))
        vis_fix_map3 = vis_fix_map3.astype(np.uint8)
        vis_fix_map3 = cv2.applyColorMap(vis_fix_map3, cv2.COLORMAP_JET)
        fin3 = cv2.addWeighted(vis_fix_map3, 0.7, orig, 0.3, 0)
        #print('orig:',vis_fix_map3.shape,orig.shape)
        
        pred32 = no_crf[i][0].detach().cpu().numpy()#.detach().cpu().numpy()   .astype(np.float32)
        #print(type(pred32))
        #pred32 = np.uint8(pred32)
        #print(pred32.shape)
        #pred32 = cv2.resize(pred32,(224,224))
        pred32 = np.uint8(normalize_im(pred32)*255)
        pred32 = cv2.resize(pred32,(224,224))
        vis_fix_map32 = cv2.resize(pred32,(orig.shape[1],orig.shape[0]))
        vis_fix_map32 = np.uint8(vis_fix_map32) #.astype(np.uint8)
        #print(np.max(vis_fix_map32),np.min(vis_fix_map32),vis_fix_map32.shape)
        vis_fix_map32 = cv2.applyColorMap(vis_fix_map32, cv2.COLORMAP_JET)
        fin32 = cv2.addWeighted(vis_fix_map32, 0.7, orig, 0.3, 0)
       
        
        pred3 = gt #.detach().cpu().numpy()
        pred3 = cv2.resize(pred3,(224,224))
        pred3 = np.uint8(normalize_im(pred3)*255)
        vis_fix_map3 = cv2.resize(pred3,(orig.shape[1],orig.shape[0]))
        vis_fix_map3 = vis_fix_map3.astype(np.uint8)
        vis_fix_map3 = cv2.applyColorMap(vis_fix_map3, cv2.COLORMAP_JET)
        fin35 = cv2.addWeighted(vis_fix_map3, 0.8, orig, 0.2, 0)
        
        #print(self_att_wts.shape)
        # pred3 = self_att_wts[i].detach().cpu().numpy()
        # pred3 = cv2.resize(pred3,(224,224))
        # pred3 = np.uint8(normalize_im(pred3)*255)
        # vis_fix_map3 = cv2.resize(pred3,(orig.shape[1],orig.shape[0]))
        # vis_fix_map3 = vis_fix_map3.astype(np.uint8)
        # vis_fix_map3 = cv2.applyColorMap(vis_fix_map3, cv2.COLORMAP_JET)
        # fin36 = cv2.addWeighted(vis_fix_map3, 0.8, orig, 0.2, 0)
        
        
        fin_comb = np.concatenate((orig,fin3),0)
        fin_comb = np.concatenate((fin_comb,fin32),0)
        #fin_comb = np.concatenate((fin_comb,fin36),0)
        fin_comb = np.concatenate((fin_comb,fin35),0)
        
        
        if i == 0:
            fin1 = fin_comb
        else:
            fin1 = np.concatenate((fin1,fin_comb),1)
            
    fin1 = cv2.resize(fin1,(int(fin1.shape[1]/2),int(fin1.shape[0]/2)))
    cv2.imwrite('/home/schakraborty/cvpr_2023/code_review/vis_sam_preds/'+str(idx)+'_2.png',fin1)

    
def visualize_preds2(preds,attwts,self_attwts,attwts_mod,boxes,points_all,init_masks,masks_pred,root1,pths,epoch,idx):

    for i in range(10): 
        orig = cv2.imread(root1+'/'+pths[i][0])
        #point = points_all[i]
        
        pred = preds[i][0].detach().cpu().numpy()
        pred = np.uint8(normalize_im(pred)*255)
        vis_fix_map = cv2.resize(pred,(orig.shape[1],orig.shape[0]))
        vis_fix_map = cv2.applyColorMap(vis_fix_map, cv2.COLORMAP_JET)
        fin = cv2.addWeighted(vis_fix_map, 0.7, orig, 0.3, 0)
        
        pred2 = init_masks[i][0]#.detach().cpu().numpy()
        pred2 = np.uint8(normalize_im(pred2)*255)
        vis_fix_map2 = cv2.resize(pred2,(orig.shape[1],orig.shape[0]))
        vis_fix_map2 = vis_fix_map2.astype(np.uint8)
        vis_fix_map2 = cv2.applyColorMap(vis_fix_map2, cv2.COLORMAP_JET)
        fin2 = cv2.addWeighted(vis_fix_map2, 0.7, orig, 0.3, 0)
        
        pred3 = attwts_mod[i][0].detach().cpu().numpy()
        pred3 = np.uint8(normalize_im(pred3)*255)
        vis_fix_map3 = cv2.resize(pred3,(orig.shape[1],orig.shape[0]))
        vis_fix_map3 = vis_fix_map3.astype(np.uint8)
        vis_fix_map3 = cv2.applyColorMap(vis_fix_map3, cv2.COLORMAP_JET)
        fin3 = cv2.addWeighted(vis_fix_map3, 0.7, orig, 0.3, 0)
        
        pred4 = self_attwts[i][0].detach().cpu().numpy()
        pred4 = np.uint8(normalize_im(pred4)*255)
        vis_fix_map4 = cv2.resize(pred4,(orig.shape[1],orig.shape[0]))
        vis_fix_map4 = vis_fix_map4.astype(np.uint8)
        vis_fix_map4 = cv2.applyColorMap(vis_fix_map4, cv2.COLORMAP_JET)
        fin4 = cv2.addWeighted(vis_fix_map4, 0.7, orig, 0.3, 0)
        
        pred5 = masks_pred[i][0].squeeze(0).detach().cpu().numpy()
        #print(pred5.shape)
        pred5 = np.uint8(normalize_im(pred5)*255)
        vis_fix_map5 = cv2.resize(pred5,(orig.shape[1],orig.shape[0]))
        vis_fix_map5 = vis_fix_map5.astype(np.uint8)
        vis_fix_map5 = cv2.applyColorMap(vis_fix_map5, cv2.COLORMAP_JET)
        fin6 = cv2.addWeighted(vis_fix_map5, 0.7, orig, 0.3, 0)
        
        box = boxes[i][0].astype(np.uint8)
        start_point = (box[0], box[1])
        end_point = (box[2], box[3])
        color = (255, 0, 0)
        thickness = 2
        fin5 = cv2.rectangle(orig, start_point, end_point, color, thickness)
        
        radius = 5
        color = (255, 0, 0)
        thickness = 1
        #print(points)
        points = points_all[i] #.detach().cpu().numpy()
        #print(len(points))
        for j in range(len(points)):
            center_coordinates = (int(points[j][0]), int(points[j][1]))
            #print(center_coordinates)
            fin5 = cv2.circle(fin5, center_coordinates, radius, color, thickness)
            
        #fin5 = cv2.addWeighted(vis_fix_map5, 0.7, orig, 0.3, 0)

        fin_comb = np.concatenate((orig,fin4),0)
        fin_comb = np.concatenate((fin_comb,fin3),0)
        fin_comb = np.concatenate((fin_comb,fin2),0)
        fin_comb = np.concatenate((fin_comb,fin5),0)
        fin_comb = np.concatenate((fin_comb,fin),0)
        fin_comb = np.concatenate((fin_comb,fin6),0)
        
        if i == 0:
            fin1 = fin_comb
        else:
            fin1 = np.concatenate((fin1,fin_comb),1)

            
    fin1 = cv2.resize(fin1,(int(fin1.shape[1]/2),int(fin1.shape[0]/2)))
    cv2.imwrite('/home/schakraborty/cvpr_2023/code_review/vis_sam_preds3/'+str(epoch)+'_'+str(idx)+'.png',fin1)
 
def visualize_preds4(preds,gts,cross_wts,fed_mask,paths):

    for i in range(min(10,len(preds))):
        
        root1 = '/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/COCO9213/img_bilinear_224'
        #print(root1+'/'+paths[i][0])
        orig = cv2.imread(root1+'/'+paths[i][0][:-4]+'.png')
        orig = cv2.resize(orig,(224,224))
        
        pred3 = preds[i][0].detach().cpu().numpy()
        pred3 = cv2.resize(pred3,(224,224))
        pred3 = np.uint8(normalize_im(pred3)*255)
        vis_fix_map3 = cv2.resize(pred3,(orig.shape[1],orig.shape[0]))
        vis_fix_map3 = vis_fix_map3.astype(np.uint8)
        vis_fix_map3 = cv2.applyColorMap(vis_fix_map3, cv2.COLORMAP_JET)
        fin3 = cv2.addWeighted(vis_fix_map3, 0.7, orig, 0.3, 0)
       
        pred3 = cross_wts[i][0].detach().cpu().numpy()
        pred3 = cv2.resize(pred3,(224,224))
        pred3 = np.uint8(normalize_im(pred3)*255)
        vis_fix_map3 = cv2.resize(pred3,(orig.shape[1],orig.shape[0]))
        vis_fix_map3 = vis_fix_map3.astype(np.uint8)
        vis_fix_map3 = cv2.applyColorMap(vis_fix_map3, cv2.COLORMAP_JET)
        fin34 = cv2.addWeighted(vis_fix_map3, 0.7, orig, 0.3, 0)
        
        pred3 = fed_mask[i][0].detach().cpu().numpy()
        pred3 = cv2.resize(pred3,(224,224))
        pred3 = np.uint8(normalize_im(pred3)*255)
        vis_fix_map3 = cv2.resize(pred3,(orig.shape[1],orig.shape[0]))
        vis_fix_map3 = vis_fix_map3.astype(np.uint8)
        vis_fix_map3 = cv2.applyColorMap(vis_fix_map3, cv2.COLORMAP_JET)
        fin36 = cv2.addWeighted(vis_fix_map3, 0.7, orig, 0.3, 0)
        
        pred3 = gts[i][0].detach().cpu().numpy()
        pred3 = cv2.resize(pred3,(224,224))
        pred3 = np.uint8(normalize_im(pred3)*255)
        vis_fix_map3 = cv2.resize(pred3,(orig.shape[1],orig.shape[0]))
        vis_fix_map3 = vis_fix_map3.astype(np.uint8)
        vis_fix_map3 = cv2.applyColorMap(vis_fix_map3, cv2.COLORMAP_JET)
        fin35 = cv2.addWeighted(vis_fix_map3, 0.8, orig, 0.2, 0)
            
        fin_comb = np.concatenate((orig,fin3),0)
        fin_comb = np.concatenate((fin_comb,fin34),0)
        fin_comb = np.concatenate((fin_comb,fin36),0)
        fin_comb = np.concatenate((fin_comb,fin35),0)

        if i == 0:
            fin1 = fin_comb
        else:
            fin1 = np.concatenate((fin1,fin_comb),1)
            
    fin1 = cv2.resize(fin1,(int(fin1.shape[1]/2),int(fin1.shape[0]/2)))
    cv2.imwrite('/home/schakraborty/cvpr_2023/code_review/vis_sam_preds/'+paths[i][0].split('/')[0]+'.png',fin1)
    
def compute_bounding_box(segmentation_mask):
    # Step 1: Convert the segmentation mask to binary
    binary_mask = segmentation_mask > 0  # You may need to adjust the threshold if needed

    # Step 2: Find the coordinates of the foreground pixels
    foreground_pixels = torch.nonzero(binary_mask, as_tuple=True)

    if len(foreground_pixels[0]) == 0:
        # No foreground pixels found, return a default bounding box
        return (0, 0, 0, 0)

    # Step 3: Calculate the bounding box
    min_x = torch.min(foreground_pixels[1])
    max_x = torch.max(foreground_pixels[1])
    min_y = torch.min(foreground_pixels[0])
    max_y = torch.max(foreground_pixels[0])

    # The bounding box is defined by (min_x, min_y, max_x, max_y)
    return (min_x.item(), min_y.item(), max_x.item(), max_y.item())


def find_closest_box(init_seg):
    return compute_bounding_box(init_seg)   
    
def self_attention_module2(x,patch_size2,model_dino):
    device = torch.device("cuda")
    for i in range(len(x)):
        img = x[i]
        img_w,img_h = img.shape[1],img.shape[2]
        w, h = img.shape[1] - img.shape[1] % patch_size2, img.shape[2] - img.shape[2] % patch_size2
        img = img[:, :w, :h].unsqueeze(0)
        w_featmap = img.shape[-2] // patch_size2
        h_featmap = img.shape[-1] // patch_size2

        class_tok,patch_toks = model_dino.forward(img.to(device))
        
        self_att_map = model_dino.get_last_selfattention(img.cuda())
        nh = self_att_map.shape[1]
        self_att_map = self_att_map[0, :, 0, 1:].reshape(nh, -1)
        self_att_map = self_att_map.reshape(nh, 28, 28)
        self_att_map = F.interpolate(self_att_map.unsqueeze(0), scale_factor=16, mode="nearest")[0] #.cpu().numpy()
        self_att_map = torch.mean(self_att_map,0)
        self_att_map = (self_att_map - torch.min(self_att_map))/(torch.max(self_att_map)-torch.min(self_att_map))


        if i == 0:
            self_attn_maps = self_att_map.unsqueeze(0)
            patch_toks_group = patch_toks
        else:
            self_attn_maps = torch.cat((self_attn_maps,self_att_map.unsqueeze(0)),0)
            patch_toks_group = torch.cat((patch_toks_group,patch_toks),0)
            
    #print('pg:',patch_toks_group.shape)
    return self_attn_maps,patch_toks_group
        

    
# def self_attention_module3(x,patch_size2):
    # device = torch.device("cuda")
    # for i in range(len(x)):
        # img = x[i]
        # img_w,img_h = img.shape[1],img.shape[2]
        # w, h = img.shape[1] - img.shape[1] % patch_size2, img.shape[2] - img.shape[2] % patch_size2
        # img = img[:, :w, :h].unsqueeze(0)
        # xp = img.to(device)
        
        # x1 = backbone.conv1(xp)
        # x2 = backbone.conv2(x1)
        # x3 = backbone.conv3(x2)
        # x4 = backbone.conv4(x3)
        # x5 = backbone.conv5(x4)
    
        # patch_toks = x5 #model_dino(img.to(device))
        ##print(img.shape,patch_toks.shape)
        # patch_toks = torch.permute(patch_toks,(0,2,3,1))
        # patch_toks = torch.reshape(patch_toks,(patch_toks.shape[0],14*14,512))
        # if i == 0:
            # patch_toks_group = patch_toks
        # else:
            # patch_toks_group = torch.cat((patch_toks_group,patch_toks),0)
    # return patch_toks_group
    
    
# def conv_desc():
    # model1 = models.vgg16(pretrained=True)
    # vgg16_features = model1.features #nn.Sequential(*list(model1.children())[:-3])
    # model1 = nn.Sequential(*list(vgg16_features.children())[:-2])
    # for p in model1.parameters():
        # p.requires_grad = False
    # model1.eval()    
    # device = torch.device("cuda")
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # model1.to(device)
    # return model1 #model_without_fc #,hook_handle

    
def dino_desc():
    model1 = vits.__dict__['vit_base'](patch_size=8, num_classes=0) # no grad
    for p in model1.parameters():
        p.requires_grad = False
    model1.eval()
    device = torch.device("cuda")
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model1.to(device)
    state_dict = torch.load('/home/schakraborty/cvpr_2023/code_review/dino_vitbase8_pretrain.pth') 
    model1.load_state_dict(state_dict, strict=True)
    return model1
    


    
    
def get_saliency(fg_wts, attn_maps):
    patch_feats = attn_maps.clone()
    for i in range(len(patch_feats)):
        x1 = patch_feats[i].reshape((1,28*28)).cuda()
        y1_fg = fg_wts[i].cuda()
        comb = x1*y1_fg
        if i == 0:
            fg_sal = torch.mean(comb) #/torch.numel(x1[x1>0.15])
        else:
            fg_sal += torch.mean(comb) #/torch.numel(x1[x1>0.15])
            
    return 1-(fg_sal/len(patch_feats))
    
    # th_map[th_map < 0.15] = 0
    # th_map = th_map[0].unsqueeze(1)
    # avg_conf = torch.sum(th_map)/torch.numel(th_map[th_map >= 0.15])
    
    
def get_embeddings(fg_wts,patch_toks_group):
    bg_wts = 1-fg_wts
    patch_feats = patch_toks_group.clone()
    patch_feats = torch.permute(patch_feats,(0,2,1))

    for i in range(len(patch_feats)):
        x1 = patch_feats[i]
        y1_fg = fg_wts[i]
        y1_bg = bg_wts[i]

        fg_embed = torch.mean(x1*y1_fg,1)
        bg_embed = torch.mean(x1*y1_bg,1)
        
        if i == 0:
            fg_embeds = fg_embed.unsqueeze(0)
            bg_embeds = bg_embed.unsqueeze(0)
        else:
            fg_embeds = torch.cat((fg_embeds,fg_embed.unsqueeze(0)),0)
            bg_embeds = torch.cat((bg_embeds,bg_embed.unsqueeze(0)),0)
            
    return fg_embeds,bg_embeds
                

def get_embeddings_mask(fg_wts,patch_toks_group):
    bg_wts = 1-fg_wts
    patch_feats = patch_toks_group.clone()
    patch_feats = torch.permute(patch_feats,(0,2,1))

    for i in range(len(patch_feats)):
        x1 = patch_feats[i]
        y1_fg = fg_wts[i]
        y1_bg = bg_wts[i]

        fg_embed = torch.sum(x1*y1_fg,1)/torch.numel(y1_fg[y1_fg == 1])
        bg_embed = torch.sum(x1*y1_bg,1)/torch.numel(y1_fg[y1_bg == 1])
        #print(fg_embed,torch.numel(y1_fg[y1_fg == 1]))
        if i == 0:
            fg_embeds = fg_embed.unsqueeze(0)
            bg_embeds = bg_embed.unsqueeze(0)
        else:
            fg_embeds = torch.cat((fg_embeds,fg_embed.unsqueeze(0)),0)
            bg_embeds = torch.cat((bg_embeds,bg_embed.unsqueeze(0)),0)
            
    return fg_embeds,bg_embeds

def dense_crf(img, init_seg, sxy_gaussian,compat_gaussian,sxy_bilateral,srgb_bilateral,compat_bilateral):
    c = init_seg.shape[0]
    h = init_seg.shape[1]
    w = init_seg.shape[2]

    d = dcrf.DenseCRF2D(w, h, c)
    
    U = unary_from_softmax(init_seg)

    U = np.ascontiguousarray(U)

    d.setUnaryEnergy(U)
    
    #d.addPairwiseGaussian(sxy=20, compat=10)
    #d.addPairwiseBilateral(sxy=13, srgb=20, rgbim=img, compat=10)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=img, compat=compat_bilateral, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(10)
    Q = np.array(Q).reshape((c, h, w))
    #print(c,h,w,np.array(Q).shape)
    #Q = np.argmax(np.array(Q).reshape((2, h, w)), axis=0) 
    
    return Q
    
def dense_crf2(img, init_seg,gt_conf,sxy_gaussian,compat_gaussian,sxy_bilateral,srgb_bilateral,compat_bilateral):
    c = init_seg.shape[0]
    h = init_seg.shape[1]
    w = init_seg.shape[2]
    
    d = dcrf.DenseCRF2D(w, h, 2)
    
    U = unary_from_labels(init_seg,2,gt_conf,zero_unsure=False)

    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=img,
                           compat=compat_bilateral,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(10)
    Q = np.argmax(np.array(Q).reshape((2, h, w)), axis=0) 
    
    return Q


def apply_crf(preds_fin,paths,mode,dataset,imp_type):
    #if mode == 'test':
    for i in range(len(preds_fin)):
        if mode == 'train':
            #print('/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/COCO9213/img_bilinear_224/'+paths[i][0])
            img = cv2.imread('/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/COCO9213/img_bilinear_224/'+paths[i][0])
            #img = cv2.imread('/mnt/data10/shared/jzhang/ImageNet/ILSVRC2012/imagenet/train/'+paths[i][0])
        else:
            if dataset == 'CoCA':
                img = cv2.imread('/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/CoCA/image/'+paths[i][0][:-4]+'.jpg')
            elif dataset == 'Cosal2015':
                img = cv2.imread('/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/Cosal2015/image/'+paths[i][0][:-4]+'.jpg')
            elif dataset == 'CoSOD3k':
                img = cv2.imread('/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/CoSOD3k/image/'+paths[i][0][:-4]+'.jpg')
            elif dataset == 'coco9213':
                img = cv2.imread('/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/COCO9213/img_bilinear_224/'+paths[i][0][:-4]+'.png') 
            elif dataset == 'MSRC':
                #print('in MSRC')
                img = cv2.imread('/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/MSRC/image/'+paths[i][0][:-4]+'.bmp')
            elif dataset == 'iCoseg':
                #print('in MSRC')
                img = cv2.imread('/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/iCoseg/image/'+paths[i][0][:-4]+'.jpg')
            
            
                
        #print('/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/CoCA/image/'+paths[i][0][:-4]+'.jpg')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(224,224))
            
        U1 = preds_fin[i].detach().cpu().numpy()
        
        #U1 = (U1-np.min(U1))/(np.max(U1)-np.min(U1))
        #U2 = 1-U1
        #U = np.concatenate((U1,U2),0)
        #map1 = dense_crf(img, U)[0]
        #print(dataset)
        
        if imp_type == 'label':
            if dataset == 'CoCA':
                map1 = dense_crf2(img, U1,0.995,sxy_gaussian=10,compat_gaussian=10,sxy_bilateral=10,srgb_bilateral=3,compat_bilateral=10)
                
            if dataset == 'Cosal2015':
                map1 = dense_crf2(img, U1,0.995,sxy_gaussian=20,compat_gaussian=10,sxy_bilateral=10,srgb_bilateral=3,compat_bilateral=10)
                
            if dataset == 'CoSOD3k':
                map1 = dense_crf2(img, U1,0.995,sxy_gaussian=10,compat_gaussian=10,sxy_bilateral=10,srgb_bilateral=3,compat_bilateral=10)
        else:
            U1 = (U1-np.min(U1))/(np.max(U1)-np.min(U1))
            U2 = 1-U1
            U = np.concatenate((U1,U2),0)
            map1 = dense_crf(img, U, sxy_gaussian=13,compat_gaussian=3,sxy_bilateral=40,srgb_bilateral=13,compat_bilateral=10)
        
        #print('map1:',map1.shape,np.unique(map1))
        map1 = cv2.resize(map1,(224,224))
        map1 = (map1-np.min(map1))/(np.max(map1)-np.min(map1))
        map1 = np.expand_dims(map1, axis=0)
        
        if i == 0:
            maps = map1 
        else:
            maps = np.concatenate((maps,map1))
                
    preds_fin = torch.from_numpy(maps).cuda()
    
    return preds_fin
    
def rgb2hsv_torch(rgb):
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv):
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb
    

    
def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0,1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    return A, D

def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    return eigenvec, second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc
    
def do_ncut(feats):
    #print('feats:',feats.shape)
    # construct the affinity matrix
    A, D = get_affinity_matrix(feats, 0.15)
    # get the second smallest eigenvector
    eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
    # get salient area
    #print('eigenvec, second_smallest_vec:',eigenvec.shape, second_smallest_vec.shape)
    bipartition = get_salient_areas(second_smallest_vec)
    #print('bipart:',bipartition.shape)
    #bipartition = bipartition.reshape([16,28,28])
    #print('bipart:',bipartition.shape)
    
    # check if we should reverse the partition based on:
    # 1) peak of the 2nd smallest eigvec 2) object centric bias
    
    #seed = np.argmax(np.abs(second_smallest_vec))
    for i in range(0,int(feats.shape[1]/784)):
        second_smallest_vec1 = second_smallest_vec[i*784:(i+1)*784]
        #print(second_smallest_vec1.shape)
        seed1 = np.argmax(np.abs(second_smallest_vec1))
        bipartition1 = bipartition[i*784:(i+1)*784]
        eigenvec1 = eigenvec[i*784:(i+1)*784]
        nc = check_num_fg_corners(bipartition1, [28,28])
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition1[seed1] != 1

        if reverse:
            #reverse bipartition, eigenvector and get new seed
            eigenvec1 = eigenvec1 * -1
            bipartition1 = np.logical_not(bipartition1)
            seed1 = np.argmax(eigenvec1)
        else:
            seed1 = np.argmax(second_smallest_vec1)

        ##get pixels corresponding to the seed
        dims = [28,28]
        bipartition1 = bipartition1.reshape(dims).astype(float)
        _, _, _, cc = detect_box(bipartition1, seed1, dims, scales=[8,8], initial_im_size=[224,224])
        pseudo_mask = np.zeros(dims)
        pseudo_mask[cc[0],cc[1]] = 1
        pseudo_mask = torch.from_numpy(pseudo_mask)
        pseudo_mask = pseudo_mask.to('cuda')
        #ps = pseudo_mask.shape[0]
        #print(pseudo_mask.shape)
        if i == 0:
            pseudo_masks = pseudo_mask.unsqueeze(0) #np.expand_dims(bipartition1, axis=0)  #pseudo_mask
        else:
            pseudo_masks = torch.cat((pseudo_masks,pseudo_mask.unsqueeze(0)),0) #np.expand_dims(bipartition1, axis=0)
        
    pseudo_masks = pseudo_masks.detach().cpu().numpy()
    #print(pseudo_masks.shape)        
    return pseudo_masks #seed, bipartitions, eigvecs