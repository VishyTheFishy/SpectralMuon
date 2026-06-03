"""
airbench94_muon.py
Runs in 2.59 seconds on a 400W NVIDIA A100 using torch==2.4.1
Attains 94.01 mean accuracy (n=200 trials)
Descends from https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""
#############################################
#                 Setup                     #
#############################################

import os
import random
import sys
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# Capture code for logging if needed
with open(sys.argv[0]) as f:
    code = f.read()

torch.backends.cudnn.benchmark = True

#############################################
#               Muon optimizer              #
#############################################

@torch.compile
def zeropower_via_newtonschulz5(G, steps=7, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3, -16/5, 6/5)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
def targeted_newtonschulz5(G, steps:int = 7, tau: float = 1., return_top: bool = True):
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-1) > G.size(-2):
        X = X.mT
    n = X.size(-1)
    I = torch.eye(n, dtype=X.dtype, device=X.device) 
    M = X.mT @ X - tau**2 * I
    signedM = zeropower_via_newtonschulz5(M, steps)
    projBot = 0.5 * (I - signedM)
    projTop = 0.5 * (I + signedM)
    nsX = zeropower_via_newtonschulz5(X, steps)
    nsBot = nsX @ projBot + X @ projTop
    nsTop = nsX @ projTop + X @ projBot
    if G.size(-1) > G.size(-2):
        nsBot = nsBot.mT
        nsTop = nsTop.mT
    return nsTop if return_top else nsBot

# project Y onto isospectral manifold tangent space at X
def project_onto_tangent_space(X: torch.Tensor, Y: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
    orig_shape = X.shape
    orig_dtype = X.dtype  # Remember if it was float16/half
    
    # Cast to float32 to prevent PyTorch CUDA SVD Half-precision crash
    X = X.float()
    Y = Y.float()
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        
    X_2d = X.reshape(X.shape[0], -1)
    Y_2d = Y.reshape(Y.shape[0], -1)
        
    m, n = X_2d.shape
    
    U, S, Vh = torch.linalg.svd(X_2d, full_matrices=True)
    V = Vh.mH 
    Y_tilde = U.mH @ Y_2d @ V 
    
    Z_tilde = torch.zeros_like(Y_tilde)
    
    S_pad_m = F.pad(S, (0, max(0, m - len(S))))
    S_pad_n = F.pad(S, (0, max(0, n - len(S))))
    
    sigma_i = S_pad_m.unsqueeze(1) 
    sigma_j = S_pad_n.unsqueeze(0) 
    is_same_sigma = torch.isclose(sigma_i, sigma_j, atol=tol)
    is_nonzero = sigma_i > tol
    
    mask_distinct = ~is_same_sigma
    mask_repeated_nonzero = is_same_sigma & is_nonzero
    Z_tilde = torch.where(mask_distinct, Y_tilde, Z_tilde)
    
    max_dim = max(m, n)
    Y_tilde_sq = F.pad(Y_tilde, (0, max_dim - n, 0, max_dim - m))
    Y_tilde_skew = 0.5 * (Y_tilde_sq - Y_tilde_sq.mH)
    Y_tilde_skew = Y_tilde_skew[:m, :n]
    
    Z_tilde = torch.where(mask_repeated_nonzero, Y_tilde_skew, Z_tilde)
    Z_tilde.diagonal().fill_(0)
    
    result = (U @ Z_tilde @ Vh).reshape(orig_shape)
    
    # Cast back to the original datatype (e.g. float16) before returning
    return result.to(orig_dtype)

class TrackedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, svd_prob=0):
        track_svd_this_step = random.random() < svd_prob

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group.get("weight_decay", 0.0)
            
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                
                state = self.state[p]

                if track_svd_this_step:
                    orig_g_mat = g.reshape(len(g), -1).float()
                    orig_g_spectrum = torch.linalg.svdvals(orig_g_mat)

                if weight_decay != 0:
                    g = g.add(p.data, alpha=weight_decay)

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                if track_svd_this_step:
                    p_mat = p.data.reshape(len(p.data), -1).float()
                    p_spectrum = torch.linalg.svdvals(p_mat)

                update = g 
                
                if track_svd_this_step:
                    update_mat = update.reshape(len(update), -1).float()
                    update_spectrum = torch.linalg.svdvals(update_mat)
                    
                    tangent_update = project_onto_tangent_space(p.data, update)
                    update_norm = update.norm()
                    tangent_sq = tangent_update.norm() ** 2
                    update_sq = update_norm ** 2

                    if update_sq > 0:
                        x_ratio = min((tangent_sq / update_sq).item(), 1.0)
                        eps = 1e-9
                        tangent_metric = -np.log(1.0 - x_ratio + eps)
                    else:
                        tangent_metric = 0.0


                    if "spectra_history" not in state:
                        state["spectra_history"] = {"orig_grad": [], "param": [], "update": [], "tangent_proj_percent": []}
                    
                    state["spectra_history"]["orig_grad"].append(orig_g_spectrum.detach().cpu())
                    state["spectra_history"]["param"].append(p_spectrum.detach().cpu())
                    state["spectra_history"]["update"].append(update_spectrum.detach().cpu())
                    state["spectra_history"]["tangent_proj_percent"].append(tangent_metric)    
                
                p.data.add_(update, alpha=-lr)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, targeted=False, tau=1.0, return_top=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, targeted=targeted, tau=tau, return_top=return_top)
        super().__init__(params, defaults)

    def step(self, svd_prob=0):
        track_svd_this_step = random.random() < svd_prob

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                
                state = self.state[p]

                if track_svd_this_step:
                    orig_g_mat = p.grad.reshape(len(p.grad), -1).float()
                    orig_g_spectrum = torch.linalg.svdvals(orig_g_mat)

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm())
                
                if track_svd_this_step:
                    p_mat = p.data.reshape(len(p.data), -1).float()
                    p_spectrum = torch.linalg.svdvals(p_mat)

                if group["targeted"]:
                    update = targeted_newtonschulz5(g.reshape(len(g), -1), steps=7, tau=group["tau"], return_top=group["return_top"]).view(g.shape)
                else:
                    update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) 
                
                if track_svd_this_step:
                    update_mat = update.reshape(len(update), -1).float()
                    update_spectrum = torch.linalg.svdvals(update_mat)
                    
                    tangent_update = project_onto_tangent_space(p.data, update)
                    update_norm = update.norm()
                    tangent_sq = tangent_update.norm() ** 2
                    update_sq = update_norm ** 2

                    if update_sq > 0:
                        x_ratio = min((tangent_sq / update_sq).item(), 1.0)
                        eps = 1e-9
                        tangent_metric = -np.log(1.0 - x_ratio + eps)
                    else:
                        tangent_metric = 0.0

                    if "spectra_history" not in state:
                        state["spectra_history"] = {"orig_grad": [], "param": [], "update": [], "tangent_proj_percent": []}
                    
                    state["spectra_history"]["orig_grad"].append(orig_g_spectrum.detach().cpu())
                    state["spectra_history"]["param"].append(p_spectrum.detach().cpu())
                    state["spectra_history"]["update"].append(update_spectrum.detach().cpu())
                    state["spectra_history"]["tangent_proj_percent"].append(tangent_metric)    
                
                p.data.add_(update, alpha=-lr)

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None, skew=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]

        if skew is not None:
            selected_indices = []
            for i, cls in enumerate(self.classes):
                cls_idx = (self.labels == i).nonzero(as_tuple=True)[0]
                perm = torch.randperm(len(cls_idx), device=cls_idx.device)
                n = max(int((i+1)**(-1*skew) * len(cls_idx)), 1)
                chosen = cls_idx[perm[:n]]
                selected_indices.append(chosen)
            selected_indices = torch.cat(selected_indices)
            self.images = self.images[selected_indices]
            self.labels = self.labels[selected_indices]
        
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
            
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)
    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width,     widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)

############################################
#                Logging                   #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-"*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-"*len(print_string))

logging_columns_list = ["run   ", "epoch", "train_acc", "val_acc", "tta_val_acc", "time_seconds"]
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#                Evaluation                #
############################################

def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def train_run(run_name, opt_config, model):
    batch_size = 500
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    
    total_train_steps = ceil(8 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=norm_biases,         lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
    
    optimizer1 = TrackedSGD(param_configs, momentum=0.85, nesterov=True)
    
    if opt_config["type"] == "sgd":
        optimizer2 = TrackedSGD(filter_params, lr=0.24, momentum=0.6, nesterov=True, weight_decay=wd/0.24)
    elif opt_config["type"] == "muon":
        optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True, targeted=False)
    elif opt_config["type"] == "muon_targeted":
        optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True, 
                          targeted=True, tau=opt_config["tau"], return_top=opt_config["return_top"])

    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0
    def start_timer(): starter.record()
    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    track_svd = True
    run_spectra_data = {} 

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:]+optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            for opt in optimizers:
                if track_svd: opt.step(svd_prob=0.3)
                else: opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps: break
        stop_timer()
        
        # Aggregate spectra data
        for name, param in model.named_parameters():
            state = optimizer1.state.get(param) or optimizer2.state.get(param)
            
            # Guard against empty tracking lists
            if state and "spectra_history" in state and len(state["spectra_history"]["orig_grad"]) > 0:
                if name not in run_spectra_data:
                    run_spectra_data[name] = {"grad": [], "param": [], "update": [], "tangent_proj_percent": []}
                
                # BUGFIX: Use torch.stack instead of torch.cat to preserve the 2D [Steps, S] shape per epoch
                run_spectra_data[name]["grad"].append(torch.stack(state["spectra_history"]["orig_grad"]))
                run_spectra_data[name]["param"].append(torch.stack(state["spectra_history"]["param"]))
                run_spectra_data[name]["update"].append(torch.stack(state["spectra_history"]["update"]))
                run_spectra_data[name]["tangent_proj_percent"].extend(state["spectra_history"]["tangent_proj_percent"])
                
                for key in state["spectra_history"]: state["spectra_history"][key].clear()

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        run = run_name if epoch == 0 else None
        print_training_details(locals(), is_final_entry=False)

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    # Concat the stacked matrices across epochs into one final timeline
    for name in run_spectra_data:
        run_spectra_data[name]["grad"] = torch.cat(run_spectra_data[name]["grad"], dim=0)
        run_spectra_data[name]["param"] = torch.cat(run_spectra_data[name]["param"], dim=0)
        run_spectra_data[name]["update"] = torch.cat(run_spectra_data[name]["update"], dim=0)
        run_spectra_data[name]["tangent_proj_percent"] = torch.tensor(run_spectra_data[name]["tangent_proj_percent"])

    torch.save(run_spectra_data, f"{run_name}_spectra.pt")
    print(f"Saved run spectra to {run_name}_spectra.pt")
    return tta_val_acc

def create_and_save_plot(data, title, xlabel, ylabel, filename, color):
    """Helper function to create and save individual plots safely."""
    plt.figure(figsize=(6, 4))
    plt.plot(data, linewidth=2, color=color)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_results(run_names):
    """Generates comparison plots for the ablation runs and saves them individually."""
    # Ensure our output directory exists
    output_dir = "ablation_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample data to determine the layers
    sample_data = torch.load(f"{run_names[0]}_spectra.pt", weights_only=False)
    conv_layers = [k for k in sample_data.keys() if "conv" in k]
    
    # Ensure we actually found convolutional layers to avoid crashes
    if not conv_layers:
        conv_layers = list(sample_data.keys())
        
    # Pick the layer right in the middle of our list
    middle_layer_idx = len(conv_layers) // 2
    middle_layer = conv_layers[middle_layer_idx]
    print(f"Plotting analytics for middle layer: {middle_layer} inside '{output_dir}/'")

    for run in run_names:
        data = torch.load(f"{run}_spectra.pt", weights_only=False)
        
        weight_spectra = data[middle_layer]["param"]   # Shape: [Steps, SingularValues]
        update_spectra = data[middle_layer]["update"]  # Shape: [Steps, SingularValues]
        tangent_pct = data[middle_layer]["tangent_proj_percent"] # Shape: [Steps]
        
        # Grab first [0] and last [-1] spectra
        first_weight = weight_spectra[0].numpy()
        last_weight = weight_spectra[-1].numpy()
        first_update = update_spectra[0].numpy()
        last_update = update_spectra[-1].numpy()
        
        # Define paths and create the 5 individual plots
        base_title = f"{run}"
        base_path = os.path.join(output_dir, run)

        create_and_save_plot(first_weight, f"{base_title}\nFirst Weight Spectrum", 
                             "Singular Value Index", "Magnitude", f"{base_path}_first_weight.png", 'blue')
        
        create_and_save_plot(last_weight, f"{base_title}\nLast Weight Spectrum", 
                             "Singular Value Index", "Magnitude", f"{base_path}_last_weight.png", 'orange')
        
        create_and_save_plot(first_update, f"{base_title}\nFirst Update Spectrum", 
                             "Singular Value Index", "Magnitude", f"{base_path}_first_update.png", 'green')
        
        create_and_save_plot(last_update, f"{base_title}\nLast Update Spectrum", 
                             "Singular Value Index", "Magnitude", f"{base_path}_last_update.png", 'red')
                             
        create_and_save_plot(tangent_pct.numpy(), f"{base_title}\nLog-Barrier of Update", 
                             "Tracking Step", "Metric Value", f"{base_path}_log_barrier.png", 'purple')

    print(f"Saved all individual plots to '{output_dir}/'")

if __name__ == "__main__":
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")
    print_columns(logging_columns_list, is_head=True)
    
    train_run("warmup", {"type": "sgd"}, model)
    
    tau_values = [0.1, 0.5, 1., 5., 10.]
    configurations = []
    # 1. Tracked SVD and 2. Regular Muon
    configurations = [
        {"name": "Tracked_SGD", "config": {"type": "sgd"}},
        {"name": "Muon_Standard", "config": {"type": "muon"}}
    ]
    
    # 3. Targeted Muon Top and Bottom for n values of tau
    for tau in tau_values:
        configurations.append({
            "name": f"Muon_Targeted_Bot_tau{tau}", 
            "config": {"type": "muon_targeted", "tau": tau, "return_top": False}
        })
        configurations.append({
            "name": f"Muon_Targeted_Top_tau{tau}", 
            "config": {"type": "muon_targeted", "tau": tau, "return_top": True}
        })

    run_names = []
    for cfg in configurations:
        print(f"\n>>> Starting Run: {cfg['name']} <<<")
        train_run(cfg["name"], cfg["config"], model)
        run_names.append(cfg["name"])

    #print("\n>>> Generating Plots <<<")
    #plot_results(run_names)
