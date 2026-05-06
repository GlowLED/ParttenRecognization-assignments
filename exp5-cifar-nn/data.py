import os
import pickle
import tarfile
import numpy as np
import urllib.request
from typing import Tuple, Optional


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class CIFAR10Loader:
    """
    CIFAR-10 数据加载器
    负责下载、读取和预处理 CIFAR-10 数据集。

    数据集包含 10 个类别: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    训练集 50000 张图片，测试集 10000 张图片
    每张图片尺寸为 32x32 RGB (3072 = 32*32*3)
    """
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.base_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.archive_name = 'cifar-10-python.tar.gz'
        self.train_batches = [f'data_batch_{i}' for i in range(1, 6)]
        self.test_batch = 'test_batch'

    def download(self):
        """下载 CIFAR-10 数据集如果不存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"创建数据目录: {self.data_dir}")

        archive_path = os.path.join(self.data_dir, self.archive_name)
        extracted_dir = os.path.join(self.data_dir, 'cifar-10-batches-py')
        if not os.path.exists(archive_path):
            print(f"正在下载 CIFAR-10 数据集...")
            try:
                urllib.request.urlretrieve(self.base_url, archive_path)
                print(f"下载完成: {archive_path}")
            except Exception as e:
                print(f"下载失败 {self.base_url}: {e}")
                raise
        # 如果归档存在但未解压，则解压
        if not os.path.exists(extracted_dir):
            self._extract_archive(archive_path)

    def _extract_archive(self, archive_path: str):
        """解压 tar.gz 归档文件"""
        print("正在解压数据集...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(self.data_dir)
        print("解压完成")

    def _unpickle(self, filepath: str) -> dict:
        """读取 pickle 文件"""
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
        return data_dict

    def load_batch(self, batch_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载单个批次的数据"""
        filepath = os.path.join(self.data_dir, batch_dir)
        data_dict = self._unpickle(filepath)
        images = data_dict['data'].astype(np.float32) / 255.0
        labels = np.array(data_dict['labels'])
        return images, labels

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载所有训练和测试数据"""
        self.download()

        # 加载 5 个训练批次
        x_train_list = []
        y_train_list = []
        for batch_name in self.train_batches:
            batch_dir = os.path.join(self.data_dir, 'cifar-10-batches-py', batch_name)
            images, labels = self.load_batch(batch_dir)
            x_train_list.append(images)
            y_train_list.append(labels)

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # 加载测试批次
        test_batch_dir = os.path.join(self.data_dir, 'cifar-10-batches-py', self.test_batch)
        x_test, y_test = self.load_batch(test_batch_dir)

        return x_train, y_train, x_test, y_test

def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    将整数标签转换为 One-hot 编码
    Example: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """
    return np.eye(num_classes)[labels]

class DataLoader:
    """
    自定义数据加载器
    支持批量加载 (Batching) 和 随机打乱 (Shuffling)
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = x.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / batch_size))
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        """返回迭代器"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取下一个 batch"""
        if self.current_idx >= self.num_samples:
            raise StopIteration
            
        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        
        self.current_idx += self.batch_size
        return batch_x, batch_y

    def __len__(self):
        return self.num_batches