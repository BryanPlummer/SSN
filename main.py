from __future__ import division
import os, sys, shutil, time, random
import argparse
import warnings
import contextlib
import copy
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import (AverageMeter, RecorderMeter, time_string, convert_secs2time,
                   Cutout, Lighting, LabelSmoothingNLLLoss, RandomDataset,
                   PrefetchWrapper, fast_collate,
                   get_world_rank, get_world_size, get_local_rank,
                   initialize_dist, get_cuda_device, allreduce_tensor)
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from tqdm import tqdm
import models
import numpy as np
import random
import PIL
from sklearn.cluster import KMeans

# Ignore corrupted TIFF warnings in ImageNet.
warnings.filterwarnings('ignore', message='.*(C|c)orrupt\sEXIF\sdata.*')
# Ignore anomalous warnings from learning rate schedulers with GradScaler.
warnings.filterwarnings('ignore', message='.*lr_scheduler\.step.*')

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training script for Networks with Soft Sharing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('data_path', metavar='DPATH', type=str, help='Path to dataset')
parser.add_argument('--dataset', metavar='DSET', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'rand_imagenet'], help='Choose between CIFAR/ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='wrn', help='model architecture: ' + ' | '.join(model_names) + ' (default: shared wide resnet)')
parser.add_argument('--effnet_arch', metavar='ARCH', default=None, help='EfficientNet architecture type')
parser.add_argument('--depth', type=int, metavar='N', default=28, help='Used for wrn and densenet_cifar')
parser.add_argument('--wide', type=int, metavar='N', default=10, help='Used for growth on densenet cifar, width for wide resnet')

# Share params
parser.add_argument('--bank_size', type=int, default=8, help='Input > 0 indices maximum number of candidates considered for each layer')
parser.add_argument('--max_params', type=int, default=0, help='Input > 0 indicates maximum parameter size')
parser.add_argument('--group_share_type', type=str, default='emb', choices=['wavg', 'emb'], help='Parameter sharing type for learning groups')
parser.add_argument('--share_type', type=str, default='none', choices=['none', 'avg', 'wavg', 'emb'], help='Parameter sharing type')
parser.add_argument('--upsample_type', type=str, default='inter', choices=['none', 'wavg', 'inter', 'mask', 'repeat'], help='Type of filter upsampling type')
parser.add_argument('--upsample_window', type=int, default=1, help='Number of 3x3 windows to learn upsampling parameters for (not applicible to inter)')
parser.add_argument('--param_groups', type=int, default=1, help='Number of parameter groups')
parser.add_argument('--param_group_type', type=str, choices=['manual', 'random', 'learned', 'reload'], help='Method for generating parameter groups')
parser.add_argument('--param_group_max_params', type=int, default=5000000, help='Max parameter size for learning parameter groups')
parser.add_argument('--param_group_epochs', type=int, default=15, help='Pretraining epochs for learning parameter groups')
parser.add_argument('--param_group_schedule', type=int, nargs='+', default=[8, 13], help='Learning rate schedule for learning parameter groups')
parser.add_argument('--param_group_gammas', type=int, nargs='+', default=[0.1, 0.1], help='Learning rate drop for learning parameter groups')
parser.add_argument('--param_group_upsample_type', type=str, default='inter', choices=['inter', 'linear', 'mask', 'tile', 'repeat'], help='Type of filter upsampling for learning parameter groups')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--drop_last', default=False, action='store_true', help='Drap last small batch')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--no_nesterov', default=False, action='store_true', help='Disable Nesterov momentum')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'rmsproptf'],
                    help='Optimization algorithm (default: SGD)')

# default params used for swrn
parser.add_argument('--schedule', type=int, nargs='+', default=None, help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=None, help='LR is multiplied by gamma on schedule')
parser.add_argument('--warmup_epochs', type=int, default=None, help='Use a linear warmup')
parser.add_argument('--base_lr', type=float, default=0.1, help='Starting learning rate for warmup')
# Step-based schedule used for EfficientNets.
parser.add_argument('--step_size', type=int, default=None, help='Step size for StepLR')
parser.add_argument('--step_gamma', type=float, default=None, help='Decay rate for StepLR')
parser.add_argument('--step_warmup', type=int, default=None, help='Number of warmup steps')

#Regularization
# default for swrn
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--no_bn_decay', default=False, action='store_true', help='No weight decay on batchnorm')
parser.add_argument('--cutout', dest='cutout', action='store_true', help='Enable cutout augmentation')
parser.add_argument('--ema_decay', type=float, default=None, help='Elastic model averaging decay')

# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='Print frequency, minibatch-wise (default: 200)')
parser.add_argument('--save_path', type=str, default='./snapshots/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on test set')
parser.add_argument('--best_loss', default=False, action='store_true', help='Checkpoint best val loss instead of accuracy (default: no)')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--dist', default=False, action='store_true', help='Use distributed training (default: no)')
parser.add_argument('--amp', default=False, action='store_true', help='Use automatic mixed precision (default: no)')
parser.add_argument('--no_dp', default=False, action='store_true', help='Disable using DataParallel (default: no)')

# Random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--job-id', type=str, default='')

args = parser.parse_args()
args.use_cuda = (args.ngpu > 0 or args.dist) and torch.cuda.is_available()
if args.dist:
    import apex

# Handle mixed precision and backwards compatability.
if not hasattr(torch.cuda, 'amp') or not hasattr(torch.cuda.amp, 'autocast'):
    if args.amp:
        raise RuntimeError('No AMP support detected')
    # Provide dummy versions.

    def autocast(enabled=False):
        return contextlib.nullcontext()

    class GradScaler:
        def __init__(self, enabled=False):
            pass

        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, d):
            pass
else:
    from torch.cuda.amp import autocast, GradScaler

job_id = args.job_id
args.save_path = args.save_path + job_id
result_png_path = './results/' + job_id + '.png'
if not os.path.isdir('results') and get_world_rank() == 0:
    os.mkdir('results')

if get_world_rank() == 0:
    print(str(args))

if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

if args.dist:
    initialize_dist(f'./init_{args.job_id}')

best_acc = 0
best_los = float('inf')


def load_dataset():
    if args.dataset == 'cifar10':
        mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100
        num_classes = 100
    elif args.dataset not in ['imagenet', 'rand_imagenet']:
        assert False, "Unknown dataset : {}".format(args.dataset)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        #train_transform = transforms.Compose([transforms.Scale(256), transforms.RandomHorizontalFlip(), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
        if args.cutout: train_transform.transforms.append(Cutout(n_holes=1, length=16))
        #test_transform = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        # Ensure only one rank downloads
        if args.dist and get_world_rank() != 0:
            torch.distributed.barrier()

        if args.evaluate:
            train_data = dataset(args.data_path, train=True,
                                 transform=train_transform, download=True)
            test_data = dataset(args.data_path, train=False,
                                transform=test_transform, download=True)

            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        else:
            # partition training set into two instead.
            # note that test_data is defined using train=True
            train_data = dataset(args.data_path, train=True,
                                 transform=train_transform, download=True)
            test_data = dataset(args.data_path, train=True,
                                transform=test_transform, download=True)

            indices = list(range(len(train_data)))
            np.random.shuffle(indices)
            split = int(0.9 * len(train_data))
            train_indices, test_indices = indices[:split], indices[split:]
            if args.dist:
                # Use the distributed sampler here.
                train_subset = torch.utils.data.Subset(
                    train_data, train_indices)
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_subset, num_replicas=get_world_size(),
                    rank=get_world_rank())
                train_loader = torch.utils.data.DataLoader(
                    train_subset, batch_size=args.batch_size,
                    sampler=train_sampler, num_workers=args.workers,
                    pin_memory=True)
                test_subset = torch.utils.data.Subset(test_data, test_indices)
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_subset, num_replicas=get_world_size(),
                    rank=get_world_rank())
                test_loader = torch.utils.data.DataLoader(
                    test_subset, batch_size=args.batch_size,
                    sampler=test_sampler, num_workers=args.workers,
                    pin_memory=True)
            else:
                train_sampler = SubsetRandomSampler(train_indices)
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=args.batch_size,
                    num_workers=args.workers, pin_memory=True,
                    sampler=train_sampler)
                test_sampler = SubsetRandomSampler(test_indices)
                test_loader = torch.utils.data.DataLoader(
                    test_data, batch_size=args.batch_size,
                    num_workers=args.workers, pin_memory=True,
                    sampler=test_sampler)

        # Let ranks through.
        if args.dist and get_world_rank() == 0:
            torch.distributed.barrier()

    elif args.dataset == 'imagenet':
        if args.dist:
            imagenet_means = [0.485, 0.456, 0.406]
            imagenet_stdevs = [0.229, 0.224, 0.225]

            # Can just read off SSDs.
            if 'efficientnet' in args.arch:
                image_size = models.efficientnet.EfficientNet.get_image_size(
                    args.effnet_arch)
                train_transform = transforms.Compose([
                    models.efficientnet.augmentations.Augmentation(
                        models.efficientnet.augmentations.get_fastautoaugment_policy()),
                    models.efficientnet.augmentations.EfficientNetRandomCrop(
                        image_size),
                    transforms.Resize((image_size, image_size),
                                      PIL.Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                ])
                test_transform = transforms.Compose([
                    models.efficientnet.augmentations.EfficientNetCenterCrop(
                        image_size),
                    transforms.Resize((image_size, image_size),
                                      PIL.Image.BICUBIC)
                ])
            else:
                # Transforms adapted from imagenet_seq's, except that color jitter
                # and lighting are not applied in random orders, and that resizing
                # is done with bilinear instead of cubic interpolation.
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop((224, 224)),
                    # transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip()])
                test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop((224, 224))])
            train_data = dset.ImageFolder(
                args.data_path + '/train', transform=train_transform)
            test_data = dset.ImageFolder(
                args.data_path + '/val', transform=test_transform)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data, num_replicas=get_world_size(),
                rank=get_world_rank())
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size, sampler=train_sampler,
                num_workers=args.workers, pin_memory=True,
                collate_fn=fast_collate, drop_last=args.drop_last)
            train_loader = PrefetchWrapper(
                train_loader, imagenet_means, imagenet_stdevs,
                Lighting(0.1,
                         torch.Tensor([0.2175, 0.0188, 0.0045]).cuda(),
                         torch.Tensor([
                             [-0.5675, 0.7192, 0.4009],
                             [-0.5808, -0.0045, -0.8140],
                             [-0.5836, -0.6948, 0.4203],
                         ]).cuda()))
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_data, num_replicas=get_world_size(),
                rank=get_world_rank())
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args.batch_size, sampler=test_sampler,
                num_workers=args.workers, pin_memory=True,
                collate_fn=fast_collate)
            test_loader = PrefetchWrapper(
                test_loader, imagenet_means, imagenet_stdevs, None)
        else:
            import imagenet_seq
            train_loader = imagenet_seq.data.Loader(
                'train', batch_size=args.batch_size, num_workers=args.workers)
            test_loader = imagenet_seq.data.Loader(
                'val', batch_size=args.batch_size, num_workers=args.workers)
        num_classes = 1000
    elif args.dataset == 'rand_imagenet':
        imagenet_means = [0.485, 0.456, 0.406]
        imagenet_stdevs = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip()])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)])
        train_data = RandomDataset((3, 256, 256), 1200000, pil=True,
                                   transform=train_transform)
        test_data = RandomDataset((3, 256, 256), 50000, pil=True,
                                  transform=test_transform)
        if args.dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data, num_replicas=get_world_size(),
                rank=get_world_rank())
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_data, num_replicas=get_world_size(),
                rank=get_world_rank())
        else:
            train_sampler = RandomSampler(train_data)
            test_sampler = RandomSampler(test_data)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)
        train_loader = PrefetchWrapper(
            train_loader, imagenet_means, imagenet_stdevs,
            Lighting(0.1,
                     torch.Tensor([0.2175, 0.0188, 0.0045]).cuda(),
                     torch.Tensor([
                         [-0.5675, 0.7192, 0.4009],
                         [-0.5808, -0.0045, -0.8140],
                         [-0.5836, -0.6948, 0.4203],
                     ]).cuda()))
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, sampler=test_sampler, collate_fn=fast_collate)
        test_loader = PrefetchWrapper(
            test_loader, imagenet_means, imagenet_stdevs, None)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    return num_classes, train_loader, test_loader


def load_model(num_classes, log, max_params, share_type, upsample_type,
               groups=None):
    print_log("=> creating model '{}'".format(args.arch), log)
    if args.arch == 'efficientnet_imagenet':
        net = models.efficientnet_imagenet(
            args.effnet_arch, share_type, upsample_type, args.upsample_window,
            args.bank_size, max_params, groups)
    else:
        net = models.__dict__[args.arch](
            share_type, upsample_type, args.upsample_window, args.depth,
            args.wide, args.bank_size, max_params, num_classes, groups)
    print_log("=> network :\n {}".format(net), log)
    if args.dist:
        net = net.to(get_cuda_device())
    else:
        net = torch.nn.DataParallel(
            net.cuda(), device_ids=list(range(args.ngpu)))
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([p.numel() for p in trainable_params])
    print_log("Number of parameters: {}".format(params), log)
    return net


def learn_parameter_groups(train_loader, state, num_classes, log):
    print_log('Pretraining to learn parameter groups', log)
    net = load_model(num_classes, log, args.param_group_max_params,
                     args.group_share_type, args.param_group_upsample_type)
    if net.bank._layer_count <= args.param_groups:
        return list(range(net.bank._layer_count))

    if args.label_smoothing > 0.0:
        criterion = LabelSmoothingNLLLoss(args.label_smoothing).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    decay_skip = ['coefficients']
    if args.no_bn_decay:
        decay_skip.append('bn')
    params = group_weight_decay(net, state['decay'], decay_skip)
    if args.optimizer == 'sgd':
        if args.dist:
            optimizer = apex.optimizers.FusedSGD(
                params, state['learning_rate'], momentum=state['momentum'],
                nesterov=(not args.no_nesterov and state['momentum'] > 0.0))
        else:
            optimizer = torch.optim.SGD(
                params, state['learning_rate'], momentum=state['momentum'],
                nesterov=(not args.no_nesterov and state['momentum'] > 0.0))
    else:
        optimizer = models.efficientnet.RMSpropTF(
            params, state['learning_rate'], alpha=0.9, eps=1e-3,
            momentum=state['momentum'])
    if args.step_size:
        if args.param_group_schedule:
            raise ValueError('Cannot combine regular and step schedules')
        step_scheduler = torch.optim.lr_scheduler.StepLR(
           optimizer, args.step_size, args.step_gamma)
        if args.step_warmup:
            step_scheduler = models.efficientnet.GradualWarmupScheduler(
                optimizer, multiplier=1.0, warmup_epoch=args.step_warmup,
                after_scheduler=step_scheduler)
    else:
        step_scheduler = None
    if args.dist:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=True)
    train_los = -1
    num_warmup = args.param_group_epochs
    schedule = args.param_group_schedule
    gammas = args.param_group_gammas
    scaler = GradScaler(enabled=args.amp)
    for epoch in tqdm(range(num_warmup), desc='pretraining steps',
                      total=num_warmup, disable=get_world_rank() != 0):
        if not step_scheduler:
            adjust_learning_rate(optimizer, epoch, gammas, schedule, train_los)
        if args.dist:
            train_loader.sampler.set_epoch(epoch)
        _, train_los = train(
            train_loader, net, criterion, optimizer, scaler, epoch, log,
            step_scheduler)
        torch.cuda.synchronize()
    torch.distributed.barrier()

    coefficients = []
    for name, param in net.named_parameters():
        if (sum([pattern in name for pattern in ['coefficients']]) > 0
            and sum([pattern in name for pattern in ['mask']]) == 0):
            coefficients.append(param.data)

    coefficients = torch.stack(coefficients).cpu().numpy()
    kmeans = KMeans(n_clusters=args.param_groups).fit(coefficients)
    del net
    return kmeans.labels_


def get_random_parameter_groups():
    if args.arch == 'swrn':
        num_layers = 29
    elif args.arch == 'swrn_imagenet':
        num_layers = 56
    else:
        raise ValueError('Do not know number of layers for arch')
    groups = np.random.randint(args.param_groups,
                               size=(num_layers - args.param_groups))
    groups = list(groups) + list(range(args.param_groups))
    np.random.shuffle(groups)
    return groups


def get_manual_parameter_groups():
    if args.arch == 'swrn':
        groups = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 9,
                  10, 11, 12, 12, 12, 12, 12, 12, 13]
    elif args.arch == 'swrn_imagenet':
        groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 12, 13, 14,
                  15, 16, 14, 15, 16, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                  21, 22, 23, 21, 22, 23, 21, 22, 23, 21, 22, 23, 24, 25, 26,
                  27, 28, 29, 30, 28, 29, 30, 31]
    else:
        raise ValueError('Do not know manual groups for arch')
    return groups


def get_parameter_groups(train_loader, state, num_classes, log):
    if args.param_group_type == 'manual':
        return get_manual_parameter_groups()
    if args.param_group_type == 'random':
        return get_random_parameter_groups()
    if args.param_group_type == 'learned':
        return learn_parameter_groups(train_loader, state, num_classes, log)
    if args.param_group_type == 'reload':
        groups = np.load(os.path.join(
            args.save_path, 'groups.npy'))
        assert len(set(groups)) == args.param_groups
        return groups
    raise ValueError(
        f'Unknown parameter group type {args.param_group_type}')


def main():
    global best_acc, best_los

    if get_world_rank() == 0:
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        log = open(os.path.join(
            args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    else:
        log = None
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)
    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)
    print_log(f'Ranks: {get_world_size()}', log)
    print_log(f'Global batch size: {args.batch_size*get_world_size()}', log)

    if get_world_rank() == 0 and not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    num_classes, train_loader, test_loader = load_dataset()
    groups = args.param_groups
    if args.param_groups > 1:
        fn = os.path.join(args.save_path, 'groups.npy')
        if args.evaluate or args.resume:
            groups = np.load(fn)
            assert len(set(groups)) == args.param_groups
        else:
            groups = get_parameter_groups(train_loader, state, num_classes, log)
            if args.param_group_type != 'reload' and get_world_rank() == 0:
                np.save(fn, groups)
            if args.param_group_type == 'learned':
                print_log('Must restart after learning parameter groups', log)
                return
            if args.param_group_type == 'random':
                # Need to load this from rank 0 to get consistent view.
                torch.distributed.barrier()
                if get_world_rank() != 0:
                    groups = np.load(fn)
        print_log('groups- ' + ', '.join(
            [str(i) + ':' + str(g) for i, g in enumerate(groups)]), log)

    net = load_model(num_classes, log, args.max_params, args.share_type,
                     args.upsample_type, groups=groups)

    if args.label_smoothing > 0.0:
        criterion = LabelSmoothingNLLLoss(args.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    decay_skip = ['coefficients']
    if args.no_bn_decay:
        decay_skip.append('bn')
    params = group_weight_decay(net, state['decay'], decay_skip)
    if args.optimizer == 'sgd':
        if args.dist:
            optimizer = apex.optimizers.FusedSGD(
                params, state['learning_rate'], momentum=state['momentum'],
                nesterov=(not args.no_nesterov and state['momentum'] > 0.0))
        else:
            optimizer = torch.optim.SGD(
                params, state['learning_rate'], momentum=state['momentum'],
                nesterov=(not args.no_nesterov and state['momentum'] > 0.0))
    else:
        optimizer = models.efficientnet.RMSpropTF(
            params, state['learning_rate'], alpha=0.9, eps=1e-3,
            momentum=state['momentum'])

    if args.step_size:
        if args.schedule:
            raise ValueError('Cannot combine regular and step schedules')
        step_scheduler = torch.optim.lr_scheduler.StepLR(
           optimizer, args.step_size, args.step_gamma)
        if args.step_warmup:
            step_scheduler = models.efficientnet.GradualWarmupScheduler(
                optimizer, multiplier=1.0, warmup_epoch=args.step_warmup,
                after_scheduler=step_scheduler)
    else:
        step_scheduler = None

    if args.dist:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=True)
    scaler = GradScaler(enabled=args.amp)

    if args.ema_decay:
        ema_model = copy.deepcopy(net).to(get_cuda_device())
        ema_manager = models.efficientnet.EMA(args.ema_decay)
    else:
        ema_model, ema_manager = None, None

    recorder = RecorderMeter(args.epochs)
    if args.resume:
        if args.resume == 'auto':
            args.resume = os.path.join(args.save_path, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(
                args.resume,
                map_location=get_cuda_device() if args.ngpu else 'cpu')
            recorder = checkpoint['recorder']
            recorder.refresh(args.epochs)
            args.start_epoch = checkpoint['epoch']
            # Hack to load models that were wrapped in (D)DP.
            if args.no_dp:
                net = torch.nn.DataParallel(net, device_ids=[get_local_rank()])
            net.load_state_dict(checkpoint['state_dict'])
            if args.no_dp:
                net = net.module
            optimizer.load_state_dict(checkpoint['optimizer'])
            if step_scheduler:
                step_scheduler.load_state_dict(checkpoint['scheduler'])
            if ema_manager is not None:
                ema_manager.shadow = checkpoint['ema']
            if args.amp:
                scaler.load_state_dict(checkpoint['amp'])
            best_acc = recorder.max_accuracy(False)
            print_log(
                "=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(
                    args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log(
                "=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        if get_world_size() > 1:
            raise RuntimeError('Do not validate with distributed training')
        validate(test_loader, net, criterion, log)
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    for epoch in range(args.start_epoch, args.epochs):
        if step_scheduler:
            current_learning_rate = step_scheduler.get_last_lr()[0]
        else:
            current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule, train_los)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        if args.dist:
            train_loader.sampler.set_epoch(epoch)
            test_loader.sampler.set_epoch(epoch)
        train_acc, train_los = train(train_loader, net, criterion, optimizer,
                                     scaler, epoch, log, step_scheduler,
                                     ema_manager)
        torch.cuda.synchronize()

        val_acc, val_los = validate(test_loader, net, criterion, log,
                                    ema_model, ema_manager)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        is_best = False
        if args.best_loss:
            if val_los < best_los:
                is_best = True
                best_los = val_los
        else:
            if val_acc > best_acc:
                is_best = True
                best_acc = val_acc

        if get_world_rank() == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
                'scheduler': step_scheduler.state_dict() if step_scheduler else None,
                'ema': ema_manager.state_dict() if ema_manager is not None else None,
                'amp': scaler.state_dict() if args.amp else None
            }, is_best, args.save_path, 'checkpoint.pth.tar')

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if get_world_rank() == 0:
            recorder.plot_curve(result_png_path)

    if get_world_rank() == 0:
        log.close()


def train(train_loader, model, criterion, optimizer, scaler, epoch, log,
          step_scheduler=None, ema_manager=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with autocast(enabled=args.amp):
            output = model(input)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        if args.dist:
            reduced_loss = allreduce_tensor(loss.data)
            reduced_prec1 = allreduce_tensor(prec1)
            reduced_prec5 = allreduce_tensor(prec5)
            losses.update(reduced_loss.item(), input.size(0))
            top1.update(reduced_prec1.item(), input.size(0))
            top5.update(reduced_prec5.item(), input.size(0))
        else:
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step_scheduler:
            step_scheduler.step()
        optimizer.zero_grad()

        if ema_manager is not None:
            ema_manager.update(model, i)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    if args.dist:
        torch.distributed.barrier()
    torch.cuda.synchronize()
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log,
             ema_model=None, ema_manager=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if ema_model is not None:
        ema_model.module.load_state_dict(ema_manager.state_dict())
        model = ema_model

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with autocast(enabled=args.amp):
                output = model(input)
                loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            if args.dist:
                reduced_loss = allreduce_tensor(loss.data)
                reduced_prec1 = allreduce_tensor(prec1)
                reduced_prec5 = allreduce_tensor(prec5)
                losses.update(reduced_loss.item(), input.size(0))
                top1.update(reduced_prec1.item(), input.size(0))
                top5.update(reduced_prec5.item(), input.size(0))
            else:
                losses.update(loss.data.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

    print_log('  **Test**  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss {losses.avg:.5f} '.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses), log)
    return top1.avg, losses.avg


def print_log(print_string, log):
    if get_world_rank() != 0:
        return  # Only print on rank 0.
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    if get_world_rank() != 0:
        return
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule, loss):
    if not args.warmup_epochs and not schedule:
        return args.learning_rate  # Bail out here.
    if args.warmup_epochs is not None and epoch <= args.warmup_epochs:
        incr = (args.learning_rate - args.base_lr) / args.warmup_epochs
        lr = args.base_lr + incr*epoch
    else:
        lr = args.learning_rate
        assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def group_weight_decay(net, weight_decay, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue
        if sum([pattern in name for pattern in skip_list]) > 0: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
