from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import torch

# mae的预训练dataloder
class MaeCustomUnlabeledDataset(Dataset):
    def __init__(self, data_dir, base_transform):
        self.data_dir = data_dir
        self.transform = base_transform
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff') # 添加你需要支持的格式
        # 遍历 supported_formats 元组中的每个元素来检查文件名 f 是否以其中任何一个结尾。
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(self.supported_formats)]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
# mae的微调dataloader
class MaeCustomDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.label_file = label_file
        self.transform = transform
        self.target_transform = target_transform
        
        # 读取标签文件（Excel表格）
        self.labels_df = pd.read_excel(label_file)
        self.img_names = self.labels_df['图片名'].values  # 假设Excel里有图片名列存放图像文件名
        self.labels = self.labels_df.drop('图片名', axis=1).values  # 假设其他列为标签列

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path)  # 读取图片
        label = torch.tensor(self.labels[idx], dtype=torch.float)  # 标签转换为float类型
        
        if self.transform:
            image = self.transform(image)

        return image, label