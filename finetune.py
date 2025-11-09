import argparse
import datetime
import os
import time
from pathlib import Path

import torch

import util.lr_decay as lrd
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit

from timm.models.layers import trunc_normal_
from losses import TotalLossWithPriorAndRegularization # 导入修改后的 AsymmetricLoss

from engine_finetune import train_one_epoch, evaluate
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
import PIL
from util import dataloader
import config
from label_graph import build_label_graph

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=200, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--label_graph_file', type=str, required=False,
                help='File containing labels used to generate the label graph')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')

    # * Finetuning params
    parser.add_argument('--finetune', default='stage1_epoch500.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=22, type=int,
                        help='number of the classification types')
    parser.add_argument('--train_data_dir', default='data/train/images', type=str,
                        help='Directory containing training images')
    parser.add_argument('--train_label_file', default='data/train/labels.xlsx', type=str,
                        help='Path to the label file corresponding to training images')
    parser.add_argument('--valid_data_dir', default='data/val/images', type=str,
                        help='Directory containing validation images')
    parser.add_argument('--valid_label_file', default='data/val/labels.xlsx', type=str,
                        help='Path to the label file corresponding to validation images')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.set_defaults(pin_mem=True)

    return parser

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:" + str(6))   #config.gpu_name
        # 每次训练计算图改动较小使用，在开始前选取较优的基础算法（比如选择一种当前高效的卷积算法）
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    train_transform = build_transform(is_train=True, args=args)
    val_transform = build_transform(is_train=False, args=args)

    dataset_train = dataloader.MaeCustomDataset(data_dir=args.train_data_dir, label_file=args.train_label_file, transform=train_transform)
    dataset_val = dataloader.MaeCustomDataset(data_dir=args.valid_data_dir, label_file=args.valid_label_file, transform=val_transform)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # ① 原始 ViT 模型加载（不变）
    vit_model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,  # 让分类头无效，避免冲突
        drop_path_rate=args.drop_path,
        global_pool='avg'
    )

    # ② 构建 GAT 标签图（你已有）
    edge_index, edge_weight, label_names = build_label_graph.build_label_graph(args.label_graph_file)

    # ③ 包装模型
    model = models_vit.ViTWithGATWrapper(
        vit_encoder=vit_model,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_labels=len(label_names),
        hidden_dim=64,
        num_heads=8
    )
    model.vit.fc_norm.requires_grad_(False)
    model.vit.head.requires_grad_(False)

    print(model.state_dict().keys())

    if args.finetune and not args.eval:
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = torch.load(args.finetune, map_location='cpu')

        # 加载 ViT 主干参数
        state_dict = model.vit.state_dict()

        # # 保存预训练模型的键和数量到文件
        # with open("pretrained_keys.txt", "w", encoding="utf-8") as f:
        #     f.write("Keys in pretrained weights:\n")
        #     for key in checkpoint_model.keys():
        #         f.write(f"{key}\n")
        #     f.write(f"\nNumber of keys in pretrained weights: {len(checkpoint_model.keys())}\n")

        # # 保存当前模型的键和数量到文件
        # with open("current_model_keys.txt", "w", encoding="utf-8") as f:
        #     f.write("Keys in current model:\n")
        #     for key in state_dict.keys():
        #         f.write(f"{key}\n")
        #     f.write(f"\nNumber of keys in current model: {len(state_dict.keys())}\n")

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model.vit, checkpoint_model)  # 注意是 model.vit
        msg = model.vit.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # 初始化 decoder 的输出层（不是 model.head 了）
        trunc_normal_(model.decoder.image_proj.weight, std=2e-5)
        trunc_normal_(model.decoder.final_mlp[0].weight, std=2e-5)
        trunc_normal_(model.decoder.final_mlp[2].weight, std=2e-5)  # 初始化第二层的权重

    with open("model_arch.txt", "w") as file:
        file.write(str(model))
    model.to(DEVICE)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # build optimizer with layer-wise lr decay (lrd)
    no_weight_decay_list = model.vit.no_weight_decay().union({
        "decoder.image_proj.bias",
        "decoder.final_mlp.0.bias",
        "decoder.final_mlp.2.bias",
        "decoder.gat1.att_src.bias",
        "decoder.gat1.att_dst.bias",
        "decoder.gat1.lin.bias",
        "decoder.gat2.att_src.bias",
        "decoder.gat2.att_dst.bias",
        "decoder.gat2.lin.bias"
    })

    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
        # no_weight_decay_list=model.vit.no_weight_decay(),
        no_weight_decay_list=no_weight_decay_list,
        layer_decay=args.layer_decay
    )

    # 修改每个参数组的学习率，结合 lr_scale
    for param_group in param_groups:
        param_group['lr'] = args.lr * param_group['lr_scale']

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    # 打印每个参数的learing rate
    for param_group in optimizer.param_groups:
        print(f"Learning Rate: {param_group['lr']}")

    loss_scaler = NativeScaler()

    mixup_fn = None

    # 初始化
    # 强相关标签对
    related_pairs = [
        ("鲜红舌", "黄苔"),
        ("淡白舌", "白苔"),
        ("胖舌", "嫩舌"),
        ("齿痕舌", "润苔"),
        ("厚苔", "腐腻苔")
    ]

    label_names = [
        "淡白舌", "淡红舌", "鲜红舌", "绛红舌", "青紫舌",
        "嫩舌", "老舌", "瘦舌", "胖舌", "点刺舌", "裂纹舌", "齿痕舌",
        "无苔", "剥落苔", "薄苔", "厚苔", "润苔", "燥苔",
        "白苔", "黄苔", "灰黑苔", "腐腻苔"
    ]

    # 正类比例，按顺序填入
    prior_ratios = torch.tensor([
        0.2381, 0.5233, 0.1456, 0.0217, 0.1583,
        0.0453, 0.0461, 0.0786, 0.1319, 0.2294, 0.2153, 0.5322,
        0.0372, 0.0336, 0.6733, 0.2422, 0.4603, 0.0547,
        0.7825, 0.3244, 0.0342, 0.2794
    ])

    criterion = TotalLossWithPriorAndRegularization(
        label_names=label_names,
        prior_pos_ratios=prior_ratios,
        related_pairs=related_pairs,
        lambda_reg=0.1,
        lambda_kl=0.05,
        # best
        # lambda_reg=0.1,
        # lambda_kl=0.05,
        gamma_neg=config.gamma_neg,  # 假设 config 中已经定义了按类别的负类权重
        gamma_pos=config.gamma_pos,  # 假设 config 中已经定义了按类别的正类权重
    )

    print("criterion = %s" % str(criterion))

    if args.eval:
        evaluate(data_loader_val, model, DEVICE, len(dataset_val) ,args)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_val_loss = float('inf')  # 用来记录最好的验证集loss

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, DEVICE, epoch, loss_scaler,
            args.clip_grad, mixup_fn, train_dataset_len=len(dataset_train),
            args=args
        )

        best_val_loss = evaluate(data_loader_val, model, DEVICE, len(dataset_val), args, epoch, best_val_loss, criterion)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)