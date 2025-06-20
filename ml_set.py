import numpy as  np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F

# generic_model_all_A
# ***********************************************************************************
def data_read_all_A():
    train = pd.read_csv('./dataset/all/train.csv', header=0)
    test = pd.read_csv('./dataset/A/test.csv', header=0)
    return train, test

def one_hot_ncp_load_all_A():
    x_train_embedding = torch.tensor(np.load('./dataset/all/one_hot_ncp_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/A/one_hot_ncp_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def kmer_load_all_A():
    x_train_embedding = torch.tensor(np.load('./dataset/all/kmer_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/A/kmer_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def bert_load_all_A():
    x_train_str_embedding = torch.tensor(np.load('./dataset/all/bert_embeddings_train.npy')).to('cuda')
    x_test_str_embedding = torch.tensor(np.load('./dataset/A/bert_embeddings_test.npy')).to('cuda')
    return x_train_str_embedding, x_test_str_embedding
# ***********************************************************************************


# generic_model_all_C
# ***********************************************************************************
def data_read_all_C():
    train = pd.read_csv('./dataset/all/train.csv', header=0)
    test = pd.read_csv('./dataset/C/test.csv', header=0)
    return train, test

def one_hot_ncp_load_all_C():
    x_train_embedding = torch.tensor(np.load('./dataset/all/one_hot_ncp_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/C/one_hot_ncp_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def kmer_load_all_C():
    x_train_embedding = torch.tensor(np.load('./dataset/all/kmer_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/C/kmer_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def bert_load_all_C():
    x_train_str_embedding = torch.tensor(np.load('./dataset/all/bert_embeddings_train.npy')).to('cuda')
    x_test_str_embedding = torch.tensor(np.load('./dataset/C/bert_embeddings_test.npy')).to('cuda')
    return x_train_str_embedding, x_test_str_embedding
# ***********************************************************************************


# generic_model_all_G
# ***********************************************************************************
def data_read_all_G():
    train = pd.read_csv('./dataset/all/train.csv', header=0)
    test = pd.read_csv('./dataset/G/test.csv', header=0)
    return train, test

def one_hot_ncp_load_all_G():
    x_train_embedding = torch.tensor(np.load('./dataset/all/one_hot_ncp_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/G/one_hot_ncp_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def kmer_load_all_G():
    x_train_embedding = torch.tensor(np.load('./dataset/all/kmer_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/G/kmer_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def bert_load_all_G():
    x_train_str_embedding = torch.tensor(np.load('./dataset/all/bert_embeddings_train.npy')).to('cuda')
    x_test_str_embedding = torch.tensor(np.load('./dataset/G/bert_embeddings_test.npy')).to('cuda')
    return x_train_str_embedding, x_test_str_embedding
# ***********************************************************************************


# generic_model_all_U
# ***********************************************************************************
def data_read_all_U():
    train = pd.read_csv('./dataset/all/train.csv', header=0)
    test = pd.read_csv('./dataset/U/test.csv', header=0)
    return train, test

def one_hot_ncp_load_all_U():
    x_train_embedding = torch.tensor(np.load('./dataset/all/one_hot_ncp_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/U/one_hot_ncp_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def kmer_load_all_U():
    x_train_embedding = torch.tensor(np.load('./dataset/all/kmer_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/U/kmer_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def bert_load_all_U():
    x_train_str_embedding = torch.tensor(np.load('./dataset/all/bert_embeddings_train.npy')).to('cuda')
    x_test_str_embedding = torch.tensor(np.load('./dataset/U/bert_embeddings_test.npy')).to('cuda')
    return x_train_str_embedding, x_test_str_embedding
# ***********************************************************************************


# special_model_A
# ***********************************************************************************
def data_read_A():
    train = pd.read_csv('./dataset/A/train.csv', header=0)
    test = pd.read_csv('./dataset/A/test.csv', header=0)
    return train, test

def one_hot_ncp_load_A():
    x_train_embedding = torch.tensor(np.load('./dataset/A/one_hot_ncp_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/A/one_hot_ncp_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def kmer_load_A():
    x_train_embedding = torch.tensor(np.load('./dataset/A/kmer_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/A/kmer_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def bert_load_A():
    x_train_str_embedding = torch.tensor(np.load('./dataset/A/bert_embeddings_train.npy')).to('cuda')
    x_test_str_embedding = torch.tensor(np.load('./dataset/A/bert_embeddings_test.npy')).to('cuda')
    return x_train_str_embedding, x_test_str_embedding
# ***********************************************************************************


# special_model_C
# ***********************************************************************************
def data_read_C():
    train = pd.read_csv('./dataset/C/train.csv', header=0)
    test = pd.read_csv('./dataset/C/test.csv', header=0)
    return train, test

def one_hot_ncp_load_C():
    x_train_embedding = torch.tensor(np.load('./dataset/C/one_hot_ncp_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/C/one_hot_ncp_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def kmer_load_C():
    x_train_embedding = torch.tensor(np.load('./dataset/C/kmer_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/C/kmer_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def bert_load_C():
    x_train_str_embedding = torch.tensor(np.load('./dataset/C/bert_embeddings_train.npy')).to('cuda')
    x_test_str_embedding = torch.tensor(np.load('./dataset/C/bert_embeddings_test.npy')).to('cuda')
    return x_train_str_embedding, x_test_str_embedding
# ***********************************************************************************


# special_model_G
# ***********************************************************************************
def data_read_G():
    train = pd.read_csv('./dataset/G/train.csv', header=0)
    test = pd.read_csv('./dataset/G/test.csv', header=0)
    return train, test

def one_hot_ncp_load_G():
    x_train_embedding = torch.tensor(np.load('./dataset/G/one_hot_ncp_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/G/one_hot_ncp_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def kmer_load_G():
    x_train_embedding = torch.tensor(np.load('./dataset/G/kmer_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/G/kmer_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def bert_load_G():
    x_train_str_embedding = torch.tensor(np.load('./dataset/G/bert_embeddings_train.npy')).to('cuda')
    x_test_str_embedding = torch.tensor(np.load('./dataset/G/bert_embeddings_test.npy')).to('cuda')
    return x_train_str_embedding, x_test_str_embedding
# ***********************************************************************************


# special_model_U
# ***********************************************************************************
def data_read_U():
    train = pd.read_csv('./dataset/U/train.csv', header=0)
    test = pd.read_csv('./dataset/U/test.csv', header=0)
    return train, test

def one_hot_ncp_load_U():
    x_train_embedding = torch.tensor(np.load('./dataset/U/one_hot_ncp_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/U/one_hot_ncp_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def kmer_load_U():
    x_train_embedding = torch.tensor(np.load('./dataset/U/kmer_features_train.npy')).to('cuda')
    x_test_embedding = torch.tensor(np.load('./dataset/U/kmer_features_test.npy')).to('cuda')
    return x_train_embedding, x_test_embedding

def bert_load_U():
    x_train_str_embedding = torch.tensor(np.load('./dataset/U/bert_embeddings_train.npy')).to('cuda')
    x_test_str_embedding = torch.tensor(np.load('./dataset/U/bert_embeddings_test.npy')).to('cuda')
    return x_train_str_embedding, x_test_str_embedding
# ***********************************************************************************
