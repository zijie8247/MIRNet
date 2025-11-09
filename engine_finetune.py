import math
import sys
from typing import Iterable, Optional
import torch
import os
from timm.data import Mixup
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, train_dataset_len=0,
                    args=None):
    model.train(True)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    total_loss = 0

    # # 打印每个参数的learing rate
    # for param_group in optimizer.param_groups:
    #     print(f"Learning Rate: {param_group['lr']}")


    for data_iter_step, (samples, targets) in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # 打印每个参数的learing rate
        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']}")

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # 单个批次的损失
        print("epoch", epoch + 1, "batch", data_iter_step, "loss:", loss.detach().item())
        total_loss += loss.detach().item()
    
    print("epoch", epoch + 1, "train loss:", total_loss / train_dataset_len)
    with open(os.path.join(args.output_dir, "stage2_train_loss.txt"), "a") as f:
        f.write(str(total_loss/train_dataset_len) + "\n")


@torch.no_grad()
def evaluate(data_loader, model, DEVICE, val_dataset_len, args=None, epoch=0, best_val_loss = float('inf'), criterion=torch.nn.BCEWithLogitsLoss()):

    val_loss = 0.0
    # switch to evaluation mode
    model.eval()

    for batch, (data, target) in enumerate(data_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        pred = model(data)

        loss = criterion(pred, target)

        val_loss += loss.item()

     # 计算平均验证损失
    val_loss /= val_dataset_len
    print(f"Validation Loss at epoch {epoch}: {val_loss}")

    # 保存验证集上损失到文件
    with open(os.path.join(args.output_dir, "stage2_val_loss.txt"), "a") as f:
        f.write(f"Epoch {epoch}, Validation Loss: {val_loss}\n")
    
    # 保存验证集上loss最小的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
        print(f"Saved new best model with loss {best_val_loss}")
    
    return best_val_loss