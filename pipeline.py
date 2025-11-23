import argparse
import os
import time
import yaml
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
import torch.distributed as dist

try:
    import timm
except Exception:
    timm = None

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="/outputs_cifar100")
    args = parser.parse_args()
    return args


def load_config(path=None):
    default = {
        'dataset': 'CIFAR100',
        'data_root': '/data',
        'model': 'resnet18',
        'pretrained': False,
        'batch_size': 128,
        'epochs': 50,
        'num_workers': 8,
        'lr': 0.01,
        'optimizer': 'SGD',
        'scheduler': 'StepLR',
        'weight_decay': 1e-4,
        'early_stopping_patience': 8,
        'grad_clip': None,
        'batch_size_schedule': None, # [{'epoch':10, 'batch_size':256}]
        'device': 'cuda',
        'wandb': False,
        'seed': 42,
    }

    if path is None:
        return default
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    merged = default.copy()
    merged.update(cfg or {})
    return merged


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CachedImageFolder(Dataset):
    def __init__(self, base_dataset, transform=None, cache_size=1024):
        self.base = base_dataset
        self.transform = transform
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def _load_item_uncached(self, idx):
        item = self.base[idx]
        img, label = item[0], item[1]
        return img, label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if idx in self._cache:
            img, label = self._cache[idx]
            self._cache.move_to_end(idx)
        else:
            img, label = self._load_item_uncached(idx)
            self._cache[idx] = (img, label)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class SimpleMLP(nn.Module):
    def __init__(self, in_channels, img_size, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels * img_size * img_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(name, num_classes, pretrained=False, in_channels=3, img_size=32):
    name = name.lower()
    if timm is not None:
        try:
            model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, in_chans=in_channels)
            return model
        except Exception:
            pass

    if 'resnet18' in name:
        m = torchvision.models.resnet18(pretrained=pretrained)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if 'resnet50' in name:
        m = torchvision.models.resnet50(pretrained=pretrained)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    
    if 'mlp' in name:
        return SimpleMLP(in_channels, img_size, num_classes)
        
    raise ValueError(f'Unsupported model {name}')


# SAM optimizer see https://github.com/davda54/sam/tree/main 
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# Muon optimizer see https://github.com/KellerJordan/Muon
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


def build_dataloaders(cfg):
    ds = cfg['dataset'].upper()
    current_path = Path.cwd()
    root = os.path.join(current_path, cfg.get('data_root', '/data'))

    num_workers = cfg.get('num_workers', 4)
    batch_size = cfg.get('batch_size', 128)

    if ds == 'CIFAR10':
        num_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        trainset = CIFAR10(root=root, train=True, download=True, transform=None)
        testset = CIFAR10(root=root, train=False, download=True, transform=None)
    elif ds == 'CIFAR100':
        num_classes = 100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        trainset = CIFAR100(root=root, train=True, download=True, transform=None)
        testset = CIFAR100(root=root, train=False, download=True, transform=None)
    elif ds == 'MNIST':
        num_classes = 10
        train_tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        test_tf = train_tf
        trainset = MNIST(root=root, train=True, download=True, transform=None)
        testset = MNIST(root=root, train=False, download=True, transform=None)
    elif ds == 'OXFORDIIITPET' or ds == 'OXFORD_PET':
        num_classes = 37
        train_tf = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor()])
        test_tf = T.Compose([T.Resize((224,224)), T.ToTensor()])
        dataset = torchvision.datasets.OxfordIIITPet(root=root, download=True)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
        # dataset = ImageFolder(os.path.join(root, 'images'))
        # trainset, testset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    else:
        raise ValueError(f'Unsupported dataset {ds}')

    train_ds = CachedImageFolder(trainset, transform=train_tf)
    test_ds = CachedImageFolder(testset, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, num_classes


def get_optimizer(name, params, lr, weight_decay=0.0):
    name = name.lower()
    if name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'rmsprop' or name == 'rms':
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == 'sam':
        return SAM(params, optim.SGD, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == 'muon':
        return Muon(params, lr=lr, weight_decay=weight_decay, momentum=0.95)
    
    raise ValueError(f'Unsupported optimizer {name}')


def get_scheduler(name, optimizer, epochs, **kwargs):
    name = name.lower()

    base_opt = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
    if 'step' in name:
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(base_opt, step_size=step_size, gamma=gamma)
    if 'plateau' in name:
        return optim.lr_scheduler.ReduceLROnPlateau(base_opt, mode='max', patience=5)
    
    return None

def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_accum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_accum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    acc = correct / total
    avg_loss = loss_accum / total
    return avg_loss, acc


def train_one_epoch(model, loader, optimizer, device, epoch, cfg, writer=None, wandb_run=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for i, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        if isinstance(optimizer, SAM):
            loss.backward()
            optimizer.first_step(zero_grad=True)
            logits2 = model(xb)
            loss2 = criterion(logits2, yb)
            loss2.backward()
            optimizer.second_step()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward()
            if cfg.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == yb).sum().item()
        total += xb.size(0)

    avg_loss = running_loss / total
    avg_acc = running_correct / total
    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/acc', avg_acc, epoch)
    if wandb_run:
        wandb_run.log({'train/loss': avg_loss, 'train/acc': avg_acc, 'epoch': epoch})
    return avg_loss, avg_acc


def run_training(cfg, args):
    set_seed(cfg.get('seed', 42))

    device = torch.device(cfg.get('device', 'cpu') if torch.cuda.is_available() and cfg.get('device', 'cpu').startswith('cuda') else 'cpu')

    current_path = Path.cwd()
    out_dir = current_path / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(out_dir/'tb'))
    wandb_run = None
    if cfg.get('use_wandb', False) and WANDB_AVAILABLE:
        wandb_run = wandb.init(project='hw3-training', config=cfg, reinit=True)

    train_loader, test_loader, num_classes = build_dataloaders(cfg)

    sample_x, _ = next(iter(train_loader))
    in_channels = sample_x.shape[1]
    model = get_model(cfg.get('model', 'resnet18'), num_classes, pretrained=cfg.get('pretrained', False), in_channels=in_channels, img_size=sample_x.shape[2])
    model = model.to(device)

    optimizer = get_optimizer(cfg.get('optimizer','sgd'), model.parameters(), lr=cfg.get('lr',0.01), weight_decay=cfg.get('weight_decay',0.0))
    scheduler = get_scheduler(cfg.get('scheduler','steplr'), optimizer, cfg.get('epochs',50))

    best_acc = 0.0
    best_epoch = 0
    patience = cfg.get('early_stopping_patience', 10)

    batch_schedule = cfg.get('batch_size_schedule') or []

    for epoch in range(cfg.get('epochs', 50)):

        for entry in batch_schedule:
            if entry.get('epoch') == epoch:
                new_bs = entry.get('batch_size')
                train_loader = DataLoader(train_loader.dataset, batch_size=new_bs, shuffle=True, num_workers=cfg.get('num_workers',4))

        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, cfg, writer=writer, wandb_run=wandb_run)
        val_loss, val_acc = evaluate(model, test_loader, device)

        epoch_time = time.time() - t0

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        elif scheduler is not None:
            scheduler.step()

        writer.add_scalar('time/epoch', epoch_time, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        if wandb_run:
            wandb_run.log({'val/acc': val_acc, 'val/loss': val_loss, 'train/acc': train_acc, 'epoch': epoch, 'epoch_time': epoch_time})

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / 'best.pth')
        if epoch - best_epoch > patience:
            print(f'Early stopping at epoch {epoch}, best epoch {best_epoch} acc={best_acc:.4f}')
            break

        print(f'Epoch {epoch:03d} | train_acc {train_acc:.4f} val_acc {val_acc:.4f} time {epoch_time:.1f}s')

    print('Training finished. Best val acc:', best_acc)
    writer.close()
    if wandb_run:
        wandb_run.finish()
    return best_acc


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args.config)

    print('Configuration:')
    print(cfg)
    best_acc = run_training(cfg, args)
    print('Best accuracy achieved:', best_acc)
