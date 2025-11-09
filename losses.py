import torch
import torch.nn as nn

class AsymmetricLossClass(nn.Module):
    def __init__(self, gamma_neg=None, gamma_pos=None, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        # 使用 super().__init__() 来初始化父类
        super().__init__()

        # 如果没有提供，则设置为默认的相同权重
        self.gamma_neg = gamma_neg if gamma_neg is not None else [4] * 22  # 假设有22个类别
        self.gamma_pos = gamma_pos if gamma_pos is not None else [1] * 22  # 假设有22个类别
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # 转换为张量，方便计算
        self.gamma_neg = torch.tensor(self.gamma_neg)
        self.gamma_pos = torch.tensor(self.gamma_pos)

    def forward(self, x, y):
        """ 
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Ensure gamma tensors are on the same device as input tensors (x, y)
        device = x.device  # or y.device if you prefer
        self.gamma_neg = self.gamma_neg.to(device)
        self.gamma_pos = self.gamma_pos.to(device)

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg is not None or self.gamma_pos is not None:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class TotalLossWithPriorAndRegularization(nn.Module):
    def __init__(self, label_names, related_pairs, prior_pos_ratios, 
                 lambda_reg=1.0, lambda_kl=0.5, 
                 gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False): 
        super().__init__()

        # 基础 Asymmetric Loss（你可以自定义参数）
        self.asym_loss = AsymmetricLossClass(
            gamma_neg = gamma_neg,
            gamma_pos = gamma_pos,
            clip = clip,
            eps = eps,
            disable_torch_grad_focal_loss=disable_torch_grad_focal_loss
        )

        # 标签名与索引
        self.label_names = label_names
        self.name2idx = {name: idx for idx, name in enumerate(label_names)}

        # 标签正类先验分布 (Tensor [C])
        if isinstance(prior_pos_ratios, torch.Tensor):
            self.register_buffer("prior_pos_ratios", prior_pos_ratios.clone().detach().float())
        else:
            self.register_buffer("prior_pos_ratios", torch.tensor(prior_pos_ratios, dtype=torch.float32))


        # 标签相似度矩阵 A [C x C]
        A = torch.zeros(len(label_names), len(label_names), dtype=torch.float32)
        for a, b in related_pairs:
            i, j = self.name2idx[a], self.name2idx[b]
            A[i, j] = A[j, i] = 1.0
        self.register_buffer("label_similarity_matrix", A)

        # 权重
        self.lambda_reg = lambda_reg
        self.lambda_kl = lambda_kl

    def forward(self, preds, targets):
        """
        preds: [B, C]  - 模型输出概率（sigmoid后的）
        targets: [B, C] - 多标签 ground truth（0/1）
        """
        logits = preds                       # 模型原始输出
        probs = torch.sigmoid(preds)         # 概率值

        loss_asym = self.asym_loss(logits, targets)  # 输入 logits
        loss_reg = self.label_similarity_regularization(probs)
        loss_kl  = self.kl_divergence_prior(probs)

        total_loss = loss_asym + self.lambda_reg * loss_reg + self.lambda_kl * loss_kl
        return total_loss

    def label_similarity_regularization(self, preds):
        reg = 0.0
        for i in range(preds.size(1)):
            for j in range(preds.size(1)):
                sim = self.label_similarity_matrix[i, j]
                if sim > 0:
                    reg += sim * torch.mean((preds[:, i] - preds[:, j]) ** 2)
        return reg

    def kl_divergence_prior(self, preds, eps=1e-6):
        pred_mean = preds.mean(dim=0)  # shape: [C]
        prior = self.prior_pos_ratios.to(pred_mean.device)
        kl = prior * torch.log((prior + eps) / (pred_mean + eps))
        return kl.sum()