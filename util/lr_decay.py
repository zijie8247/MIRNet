def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    # ViT + GAT
    num_layers = len(model.vit.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        # print(n, p)
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # 写入信息到文件
    with open("param_groups.txt", "w") as f:
        f.write("Parameter Groups:\n")
        for group_name, group in param_group_names.items():
            f.write(f"\nGroup: {group_name}\n")
            f.write(f"  lr_scale: {group['lr_scale']}\n")
            f.write(f"  weight_decay: {group['weight_decay']}\n")
            f.write("  parameters:\n")
            for param_name in group['params']:
                f.write(f"    - {param_name}\n")
    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())

# Vit + GAT + classifer
def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id, supporting ViT + GATDecoder
    """

    if name.startswith('decoder'):
        return num_layers  # decoder 全部归一类

    # ViT encoder 部分保持不变
    if name in ['vit.cls_token', 'vit.pos_embed']:
        return 0
    elif name.startswith('vit.patch_embed'):
        return 0
    elif name.startswith('vit.blocks'):
        return int(name.split('.')[2]) + 1  # vit.blocks.0.xyz → 第 1 层起始
    else:
        return num_layers  # vit.norm, vit.head 等