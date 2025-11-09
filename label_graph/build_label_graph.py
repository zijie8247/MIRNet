import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from matplotlib import rcParams, font_manager

# 使用你系统中已安装的中文字体
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
my_font = font_manager.FontProperties(fname=font_path)

# 设置全局中文字体
rcParams['font.family'] = my_font.get_name()
rcParams['axes.unicode_minus'] = False  # 正确显示负号

def build_label_graph(excel_path):
    """
    读取标签Excel，构建标签共现图，结合先验知识调整，返回 PyG 格式的标签图边和权重

    参数：
        excel_path (str): 标签Excel文件路径，第一列是图片名，后面是标签列，标签为0/1

    返回：
        edge_index (torch.LongTensor): 图的边索引，形状 [2, 边数]
        edge_weight (torch.FloatTensor): 边权重，形状 [边数]
        label_names (list[str]): 标签名列表，节点顺序对应edge_index的节点编号
    """
    df = pd.read_excel(excel_path)
    labels_df = df.iloc[:, 1:]
    label_names = labels_df.columns.tolist()
    label2idx = {name: i for i, name in enumerate(label_names)}

    y = labels_df.to_numpy(dtype=int)
    
    M = np.dot(y.T, y)
    np.fill_diagonal(M, 0)

    # 打印共现矩阵统计信息
    nonzero_vals = M[np.nonzero(M)]
    print("共现矩阵 M 的统计信息:")
    print("最大值:", np.max(nonzero_vals))
    print("最小值:", np.min(nonzero_vals))
    print("平均值:", np.mean(nonzero_vals))
    print("中位数:", np.median(nonzero_vals))
    print("25分位数:", np.percentile(nonzero_vals, 25))
    print("非零元素数:", np.count_nonzero(nonzero_vals))

    # 设定共现次数阈值（动态分位数）
    threshold = np.percentile(nonzero_vals, 25)
    print(f"共现阈值设为: {threshold:.2f}")

    # 根据阈值构建邻接矩阵，并保留原始权重
    W = np.where(M >= threshold, M, 0).astype(np.float32)

    # 先验知识强化边
    strongly_correlated = [
        ('鲜红舌', '黄苔'),
        ('淡白舌', '白苔'),
        ('胖舌', '嫩舌'),
        ('齿痕舌', '润苔'),
        ('厚苔', '腐腻苔')
    ]
    for l1, l2 in strongly_correlated:
        i, j = label2idx[l1], label2idx[l2]
        # W[i, j] = W[j, i] = max(W[i, j], threshold + 10)
        W[i, j] = W[j, i] = max(nonzero_vals)

    # 互斥关系：强行去边
    mutually_exclusive = [
        ('厚苔', '薄苔'),
        ('润苔', '燥苔'),
        ('胖舌', '瘦舌'),
        ('老舌', '嫩舌')
    ]
    for l1, l2 in mutually_exclusive:
        i, j = label2idx[l1], label2idx[l2]
        W[i, j] = W[j, i] = 0.0

    # 1. 对数平滑（平缓大权重差异）
    W = np.log1p(W)
    # 2. 安全归一化：避免 0/0
    row_sums = np.sum(W, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # 避免除以 0
    W = W / row_sums
    # 无苔与其他苔质互斥
    no_coating = '无苔'
    conflicting_coatings = ['剥落苔', '薄苔', '厚苔', '润苔', '燥苔', '腐腻苔', '白苔', '黄苔', '灰黑苔']
    i_no_coating = label2idx[no_coating]
    for label in conflicting_coatings:
        j = label2idx[label]
        W[i_no_coating, j] = W[j, i_no_coating] = 0.0

    # 转换为 PyG 稀疏图格式
    adj_tensor = torch.tensor(W)
    edge_index, edge_weight = dense_to_sparse(adj_tensor)

    return edge_index, edge_weight, label_names

from torch_geometric.utils import dense_to_sparse
import networkx as nx
import matplotlib.pyplot as plt

def visualize_label_graph(edge_index, edge_weight, label_names, top_k=100, save_path="label_graph.png"):
    """
    可视化标签图：使用 networkx 绘制图并保存为图片

    参数：
        edge_index (torch.LongTensor): PyG 图的边索引
        edge_weight (torch.FloatTensor): PyG 图的边权重
        label_names (list[str]): 标签名列表
        top_k (int): 展示前 top_k 条边
        save_path (str): 图片保存路径
    """
    G = nx.Graph()

    # 添加节点
    for i, label in enumerate(label_names):
        G.add_node(i, label=label)

    # 提取 top_k 高权重边
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist(), edge_weight.tolist()))
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_k]

    # 添加边
    for i, j, w in edges:
        G.add_edge(i, j, weight=w)

    # 节点标签映射
    label_dict = {i: name for i, name in enumerate(label_names)}

    # 布局
    pos = nx.spring_layout(G, seed=42)

    # 绘图
    plt.figure(figsize=(16, 12))
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=12, font_family=my_font.get_name())
    nx.draw_networkx_edges(G, pos, width=[w * 4 for w in edge_weights], alpha=0.7)

    plt.title("标签共现图（Top {} 条边）".format(top_k), fontsize=16)
    plt.axis('off')

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存到：{save_path}")
    plt.close()

if __name__ == '__main__':
    edge_index, edge_weight, label_names = build_label_graph('labels.xlsx')
    print(f"标签数量: {len(label_names)}")
    print(f"边数: {edge_index.shape[1]}")
    print(edge_index)
    print(edge_weight)
     # 可视化图结构
    visualize_label_graph(edge_index, edge_weight, label_names, top_k=100)