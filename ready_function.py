import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from kmer import *

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed as {seed}")

def kmer_pairs(sequence):

    if len(sequence) != 41:
        raise ValueError("The RNA sequence must have a length of 41.")

    # Get the 21st nucleotide
    central_nucleotide = sequence[20]  # 21st position is at index 20 (0-based indexing)

    # Generate pairs with every other nucleotide
    #pairs = [central_nucleotide + nucleotide for nucleotide in sequence]
    pairs = [
        central_nucleotide + nucleotide
        for i, nucleotide in enumerate(sequence)
        # if i != 20  # Exclude pairing with itself
    ]

    # Return as a comma-separated string
    return ','.join(pairs)

def kemer_encoding(sequence, kmer):

    if not sequence or not isinstance(sequence, str):
        raise ValueError("sequence must be a non-empty string.")

    # Split the sequence into k-mers (comma-separated nucleotide pairs)
    sequence_split = sequence.split(',')

    # Encode the sequence using each kemer_2 mapping
    encodings = []
    for item in kmer:
        values = item["Values"]
        row_values = [values.get(aa_pair, 0) for aa_pair in sequence_split]
        encodings.append(row_values)

    return encodings

class Kmer_Dataset(Dataset):
    def __init__(self, sequences, labels, kmer):
        self.sequences = sequences
        self.labels = labels
        self.kmer = kmer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Generate input sequence from the RNA sequence
        sequence = self.sequences[idx]
        input_sequence = kmer_pairs(sequence)

        # Encode the sequence
        encodings = kemer_encoding(input_sequence, self.kmer)
        encodings_tensor = torch.tensor(encodings, dtype=torch.float32)

        # Convert the label to a tensor
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)  # Adjust dtype as needed
        return encodings_tensor, label

def generate_dna_bert_embeddings(sequences):

    # 指定本地模型路径
    local_model_path = r"/home/ys/shenpeng/1/DNABERT-2-117M"

    # 加载本地模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)

    model = model.to(device)

    embeddings = []

    max_length = 14

    for sequence in sequences:

        # 对输入序列进行编码
        inputs = tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True,
                           max_length=max_length).to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 生成嵌入向量
        with torch.no_grad():
            hidden_states = model(input_ids, attention_mask=attention_mask)[0]  # 获取最后一层隐藏状态

        hidden_states = hidden_states.squeeze(0)
        embeddings.append(hidden_states)

        # 利用 attention_mask 过滤填充部分的嵌入
        # mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        # filtered_hidden_states = hidden_states * mask_expanded  # 忽略填充位置
        # mean_embedding = filtered_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)  # 取非填充位置的均值
        # embeddings.append(mean_embedding)


    # 返回所有序列的嵌入向量
    return torch.stack(embeddings)

class DNA_BERT_Dataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.embeddings = generate_dna_bert_embeddings(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return embedding, label


class PairwiseDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2


    def __getitem__(self, index):
        x1, y1 = self.dataset1[index]
        x2, y2 = self.dataset2[index]
        return x1, x2, y1

    def __len__(self):
        return len(self.dataset1)

def read_dataset(data_file):
    df = pd.read_csv(data_file)
    seq = df['seq']
    label = df['label']
    X = Kmer_Dataset(seq, label, kmer)
    Y = DNA_BERT_Dataset(seq, label)
    dataset = PairwiseDataset(X, Y)
    return dataset




import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_from_csv(csv_path):
    """
    从CSV文件加载数据和标签
    :param csv_path: CSV文件路径
    :return: (序列列表, 标签列表)
    """
    df = pd.read_csv(csv_path)

    # 验证数据列是否存在
    if 'seq' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV文件必须包含 'seq' 和 'label' 列")

    sequences = df['seq'].tolist()
    labels = df['label'].values

    # 验证序列长度
    for idx, seq in enumerate(sequences):
        if len(seq) != 41:
            raise ValueError(f"第 {idx + 1} 行序列长度错误: 期望41，实际{len(seq)}")

    return sequences, labels


def save_features(features, filename, base_dir=r"/home/ys/SP/1_5/my_model/dataset/U"):
    """
    增强版特征存储函数(NumPy专用版)
    :param features: 要保存的数据（支持Tensor/ndarray）
    :param filename: 目标文件名（建议使用.npy后缀）
    :param base_dir: 存储根目录（自动创建）
    """
    # 创建目录结构
    feature_dir = Path(base_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    # 构建完整路径
    save_path = feature_dir / filename

    # 统一转换为NumPy数组
    if isinstance(features, torch.Tensor):
        # 处理GPU Tensor：先移动到CPU再转换
        numpy_features = features.cpu().numpy()
    elif isinstance(features, np.ndarray):
        numpy_features = features
    else:
        raise TypeError("仅支持保存Tensor或Numpy数组，当前类型为: {}".format(type(features)))

    # 保存为NumPy格式
    np.save(save_path, numpy_features)

    print(f"✅ 保存成功: {save_path} | 维度: {numpy_features.shape} | 格式: NumPy数组")


def one_hot(seq):
    """将RNA序列转换为one-hot编码"""
    a_vec = [1.0 if nucleotide == 'A' else 0.0 for nucleotide in seq]
    c_vec = [1.0 if nucleotide == 'C' else 0.0 for nucleotide in seq]
    g_vec = [1.0 if nucleotide == 'G' else 0.0 for nucleotide in seq]
    u_vec = [1.0 if nucleotide in ['U', 'T'] else 0.0 for nucleotide in seq]
    return a_vec, c_vec, g_vec, u_vec


def one_hot_ncp(seq):
    """组合 One-hot + NCP 编码，输出维度 [7 x seq_len]"""
    combined_vecs = []
    for nucleotide in seq:
        one_hot = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'U': [0, 0, 0, 1],
            'T': [0, 0, 0, 1],
        }.get(nucleotide, [0, 0, 0, 0])

        ncp = {
            'A': [1, 1, 1],
            'C': [0, 1, 0],
            'G': [1, 0, 0],
            'U': [0, 0, 1],
            'T': [0, 0, 1],
        }.get(nucleotide, [0, 0, 0])

        combined = one_hot + ncp
        combined_vecs.append(combined)

    # 返回 shape: [seq_len, 7]
    return np.array(combined_vecs, dtype=np.float32)



def process_full_pipeline(csv_path, kmer_params):
    """完整数据处理流水线（包含k-mer、DNA-BERT和one-hot编码）"""
    # 加载原始数据
    sequences, labels = load_data_from_csv(csv_path)

    # ================== 特征生成 ==================
    # 1. 生成k-mer特征
    kmer_features = []
    for seq in sequences:
        pairs = kmer_pairs(seq)
        encoding = kemer_encoding(pairs, kmer_params)
        kmer_features.append(encoding)
    kmer_features = np.array(kmer_features)  # 形状: [样本数, 特征数, 特征维度]

    # 2. 生成DNA-BERT嵌入
    bert_embeddings = generate_dna_bert_embeddings(sequences)

    # 3. 生成one-hot编码（形状: [样本数, 41, 4]）
    # one_hot_features = []
    # for seq in sequences:
    #     a, c, g, u = one_hot(seq)
    #     seq_one_hot = np.stack([a, c, g, u], axis=1).astype(np.float32)
    #     one_hot_features.append(seq_one_hot)
    # one_hot_features = np.array(one_hot_features)

    # 3. 生成 one-hot + NCP 编码（形状: [样本数, 41, 7]）
    one_hot_ncp_features = []
    for seq in sequences:
        combined = one_hot_ncp(seq)  # 输出 shape: [41, 7]
        one_hot_ncp_features.append(combined)
    one_hot_ncp_features = np.array(one_hot_ncp_features)  # [样本数, 41, 7]

    # ================== 数据存储 ==================
    save_features(kmer_features, "kmer_features_train.npy")
    save_features(bert_embeddings.cpu().numpy(), "bert_embeddings_train.npy")
    # save_features(one_hot_features, "one_hot_features_train.npy")
    save_features(one_hot_ncp_features, "one_hot_ncp_features_train.npy")
    save_features(labels, "labels.npy")  # 保存标签
# ... [其他已有函数，如load_data_from_csv, save_features等] ...

if __name__ == "__main__":
    input_csv = r"/home/ys/SP/1_5/my_model/dataset/U/train.csv"  # 确保路径正确
    kmer_params = kmer  # 替换为实际的k-mer参数
    process_full_pipeline(input_csv, kmer_params)