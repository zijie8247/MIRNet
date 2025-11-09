import math
import sys
import os
from typing import Iterable
import torch
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None, train_data_len=0, train_dataset_len=0):
    model.train(True)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    total_loss = 0

    for data_iter_step, samples in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device)

        with torch.amp.autocast('cuda'):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not torch.isfinite(loss):
            print(f"Loss is NaN or Inf at step {data_iter_step}")
            continue
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch} learning rate: {lr}")

        # 单个批次的损失
        print("epoch", epoch, "batch", data_iter_step, "loss:", loss.detach().item())
        total_loss += loss.detach().item()

    # 计算有效样本数
    num_effective_samples = train_data_len * args.batch_size

    # 一个epoch的平均batch损失
    epoch_loss = total_loss / train_data_len

    # 修改计算平均个体损失以考虑丢弃的样本
    individual_loss = total_loss / num_effective_samples
    print(f"epoch average batch loss: {epoch_loss} individual loss: {individual_loss}")
    with open(os.path.join(args.output_dir, "mae_stage1_loss.txt"), "a") as f:
        f.write(str(total_loss/train_dataset_len*args.batch_size) + " ")
        f.write(str(total_loss/train_dataset_len) + "\n")