# eval.py
import torch,argparse
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, auc  # === 新增导入
import models_vit
from label_graph import build_label_graph
import torchvision.transforms as transforms
import util.dataloader
import PIL
import torchvision.models as models
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from metric import calculate_tp_fp_fn_tn,calculate_accuracy,calculate_metrics,calculate_auc,calculate_f1,calculate_precision,calculate_recall

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

def eval(args):
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:" + str(4))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    eval_transform = build_transform(is_train=False, args=args)
        
    eval_dataset = util.dataloader.MaeCustomDataset(
        data_dir=args.data_dir, 
        label_file=args.label_file, 
        transform=eval_transform)
    
    eval_data=torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model_path = args.model_path

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

    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        all_true = []  # 用于存储所有真实标签
        all_pred = []  # 用于存储所有预测标签（原始概率）
        all_pred_binary = []  # 用于存储所有二值化后的预测标签

        for batch, (data, target) in enumerate(eval_data):
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data)

            # 使用sigmoid转换预测为概率
            pred_prob = torch.sigmoid(pred)
            # 将大于0.5的值视为1，小于等于0.5的视为0
            pred_binary = (pred_prob > 0.5).float()

            all_true.append(target)
            all_pred.append(pred_prob)  # 存储原始的预测概率
            all_pred_binary.append(pred_binary)  # 存储二值化后的预测结果

        # 合并所有batch的标签和预测结果
        all_true = torch.cat(all_true, dim=0)
        all_pred = torch.cat(all_pred, dim=0)

        all_pred_binary = torch.cat(all_pred_binary, dim=0)

        # 计算Hamming Accuracy, example-F1, micro-F1, macro-F1
        ha, ex_f1, micro_f1_score, macro_f1_score = calculate_metrics(all_true, all_pred_binary)

        # 初始化用于平均值统计的列表
        precisions = []
        recalls = []
        aucs = []
        pr_aucs = []  # === 新增列表用于存储每类PR-AUC

        # 输出并保存每个类别的所有指标
        with open(os.path.join(args.result_path, args.output_file), "w") as f:
            for i in range(22):
                # 提取第i类的真实标签和预测标签
                true_class = all_true[:, i].cpu().numpy()
                pred_class_binary = all_pred_binary[:, i].cpu().numpy()
                pred_prob_class = all_pred[:, i].cpu().numpy()

                # 计算该类别的TP, FP, FN, TN
                tp, fp, fn, tn = calculate_tp_fp_fn_tn(true_class, pred_class_binary)
                accuracy = calculate_accuracy(tp, fp, fn, tn)
                precision = calculate_precision(true_class, pred_class_binary)
                recall = calculate_recall(true_class, pred_class_binary)
                f1 = calculate_f1(true_class, pred_class_binary)
                auc_score = calculate_auc(true_class, pred_prob_class)

                # === 新增计算 PR-AUC ===
                try:
                    precision_curve, recall_curve, _ = precision_recall_curve(true_class, pred_prob_class)
                    pr_auc_score = auc(recall_curve, precision_curve)
                except Exception as e:
                    pr_auc_score = -1

                # 排除无法计算（返回为 -1）的值
                if precision != -1:
                    precisions.append(precision)
                if recall != -1:
                    recalls.append(recall)
                if auc_score != -1:
                    aucs.append(auc_score)
                if pr_auc_score != -1:
                    pr_aucs.append(pr_auc_score)

                # 输出该类别的所有指标
                f.write(f"Class {i} - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, Accuracy: {accuracy:.6f}, "
                        f"Precision: {precision:.6f}, Recall: {recall:.6f}, F1-Score: {f1:.6f}, AUC: {auc_score:.6f}, PR-AUC: {pr_auc_score:.6f}\n")
                print(f"Class {i} - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, Accuracy: {accuracy:.6f}, "
                    f"Precision: {precision:.6f}, Recall: {recall:.6f}, F1-Score: {f1:.6f}, AUC: {auc_score:.6f}, PR-AUC: {pr_auc_score:.6f}")

        # 计算排除-1之后的平均值
        def safe_mean(values):
            return np.mean(values) if len(values) > 0 else -1

        avg_precision = safe_mean(precisions)
        avg_recall = safe_mean(recalls)
        avg_auc = safe_mean(aucs)
        avg_pr_auc = safe_mean(pr_aucs)  # === 新增 macro PR-AUC

        # === 新增计算 micro PR-AUC ===
        # 把所有类的真实标签和预测概率展平为1维数组
        true_flat = all_true.cpu().numpy().ravel()
        pred_flat = all_pred.cpu().numpy().ravel()
        try:
            precision_curve_micro, recall_curve_micro, _ = precision_recall_curve(true_flat, pred_flat)
            micro_pr_auc = auc(recall_curve_micro, precision_curve_micro)
        except Exception as e:
            micro_pr_auc = -1

        print(f"Hamming Accuracy: {ha:.6f}")
        print(f"Example-F1: {ex_f1:.6f}")
        print(f"Micro-F1: {micro_f1_score:.6f}")
        print(f"Macro-F1: {macro_f1_score:.6f}")
        print(f"Average Precision: {avg_precision:.6f}")
        print(f"Average Recall: {avg_recall:.6f}")
        print(f"Average AUC: {avg_auc:.6f}")
        print(f"Macro PR-AUC: {avg_pr_auc:.6f}")  # === 新增输出
        print(f"Micro PR-AUC: {micro_pr_auc:.6f}")  # === 新增输出

        # 保存总体和平均指标到文件
        with open(os.path.join(args.result_path, args.output_file), "a") as f:
            f.write(f"Hamming Accuracy: {ha:.6f} ")
            f.write(f"Example-F1: {ex_f1:.6f} ")
            f.write(f"Micro-F1: {micro_f1_score:.6f} ")
            f.write(f"Macro-F1: {macro_f1_score:.6f}\n")
            f.write(f"Average Precision: {avg_precision:.6f} ")
            f.write(f"Average Recall: {avg_recall:.6f} ")
            f.write(f"Average AUC: {avg_auc:.6f}\n")
            f.write(f"Macro PR-AUC: {avg_pr_auc:.6f} Micro PR-AUC: {micro_pr_auc:.6f}\n")  # === 新增保存

if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    print(model_names)
    
    parser = argparse.ArgumentParser(description='test')
    # Dataset parameters
    parser.add_argument('--batch_size', default=200, type=int, help='')
    parser.add_argument('--nb_classes', default=22, type=int,
                        help='number of the classification types')
    parser.add_argument('--data_dir', default='data/images', type=str,
                        help='Directory containing evaluation images')
    parser.add_argument('--label_file', default='data/labels.txt', type=str,
                        help='Path to the label file used for evaluation')
    
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    
    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--model_path', default='model.pth', type=str,
                        help='Path to the model used for evaluation')
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--label_graph_file', type=str, required=False,
                    help='File containing labels used to generate the label graph')
    
    parser.add_argument('--result_path', default='output_dir', type=str, 
                        help='Directory to store results')
    parser.add_argument('--output_file', default='metric.txt', type=str,
                        help='Output file name for metrics (joined with result_path)')


    args = parser.parse_args()
    eval(args)
