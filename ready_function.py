import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from kmer import *
from pathlib import Path

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
        
    central_nucleotide = sequence[20]  # 21st position is at index 20 (0-based indexing)

    pairs = [
        central_nucleotide + nucleotide
        for i, nucleotide in enumerate(sequence)
        # if i != 20  # Exclude pairing with itself
    ]

    return ','.join(pairs)

def kemer_encoding(sequence, kmer):

    if not sequence or not isinstance(sequence, str):
        raise ValueError("sequence must be a non-empty string.")

    sequence_split = sequence.split(',')

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
        sequence = self.sequences[idx]
        input_sequence = kmer_pairs(sequence)

        encodings = kemer_encoding(input_sequence, self.kmer)
        encodings_tensor = torch.tensor(encodings, dtype=torch.float32)
        
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)  # Adjust dtype as needed
        return encodings_tensor, label

def generate_dna_bert_embeddings(sequences):

    local_model_path = "./DNABERT-2-117M"

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)

    model = model.to(device)

    embeddings = []

    max_length = 14

    for sequence in sequences:
        
        inputs = tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True,
                           max_length=max_length).to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            hidden_states = model(input_ids, attention_mask=attention_mask)[0]  

        hidden_states = hidden_states.squeeze(0)
        embeddings.append(hidden_states)

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

def load_data_from_csv(csv_path):
    
    df = pd.read_csv(csv_path)

    if 'seq' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV文件必须包含 'seq' 和 'label' 列")

    sequences = df['seq'].tolist()
    labels = df['label'].values

    for idx, seq in enumerate(sequences):
        if len(seq) != 41:
            raise ValueError(f"第 {idx + 1} 行序列长度错误: 期望41，实际{len(seq)}")

    return sequences, labels


def save_features(features, filename, base_dir="./dataset/U"):
    
    feature_dir = Path(base_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    save_path = feature_dir / filename

    if isinstance(features, torch.Tensor):
        numpy_features = features.cpu().numpy()
    elif isinstance(features, np.ndarray):
        numpy_features = features
    else:
        raise TypeError("仅支持保存Tensor或Numpy数组，当前类型为: {}".format(type(features)))

    np.save(save_path, numpy_features)

    print(f"✅ 保存成功: {save_path} | 维度: {numpy_features.shape} | 格式: NumPy数组")


def one_hot(seq):
    
    a_vec = [1.0 if nucleotide == 'A' else 0.0 for nucleotide in seq]
    c_vec = [1.0 if nucleotide == 'C' else 0.0 for nucleotide in seq]
    g_vec = [1.0 if nucleotide == 'G' else 0.0 for nucleotide in seq]
    u_vec = [1.0 if nucleotide in ['U', 'T'] else 0.0 for nucleotide in seq]
    
    return a_vec, c_vec, g_vec, u_vec


def one_hot_ncp(seq):
    
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

    return np.array(combined_vecs, dtype=np.float32)

def process_full_pipeline(csv_path, kmer_params):

    sequences, labels = load_data_from_csv(csv_path)
    
    kmer_features = []
    for seq in sequences:
        pairs = kmer_pairs(seq)
        encoding = kemer_encoding(pairs, kmer_params)
        kmer_features.append(encoding)
    kmer_features = np.array(kmer_features)  

    bert_embeddings = generate_dna_bert_embeddings(sequences)

    one_hot_ncp_features = []
    for seq in sequences:
        combined = one_hot_ncp(seq)  
        one_hot_ncp_features.append(combined)
    one_hot_ncp_features = np.array(one_hot_ncp_features)  

    save_features(kmer_features, "kmer_features_train.npy")
    save_features(bert_embeddings.cpu().numpy(), "bert_embeddings_train.npy")
    save_features(one_hot_ncp_features, "one_hot_ncp_features_train.npy")
    save_features(labels, "labels.npy")  


if __name__ == "__main__":
    input_csv = "./dataset/U/train.csv" 
    kmer_params = kmer  
    process_full_pipeline(input_csv, kmer_params)
