import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/citeseer/", dataset="citeseer"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    print("labels.max().item():", labels.max().item())
    print(idx_features_labels, idx_features_labels.shape, labels.shape[0])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    print(idx)
    idx_map = {j: i for i, j in enumerate(idx)}    # 枚举
    print("idx_map:", idx_map)
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.dtype(str))
    print("edges_unordered:", edges_unordered)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.dtype(str)).reshape(edges_unordered.shape)
    print("edges:", edges, edges.shape)
    # k =-1;
    # for ix in edges[:, 0]:
    #     k = k+1
    #     if ix == 'None':
    #         print("ix, k", ix, k)
    # k = -1;
    # for iy in edges[:, 1]:
    #     k = k + 1
    #     if iy == 'None':
    #         print("iy, k", iy, k)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),        # 3312*3312    adj sparse矩阵 方式存储
                        dtype=np.float32)
    print("adj:", adj)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print("adj_new:", adj)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    print("adj_nor", adj)
    idx_train = range(150)                     # 140
    idx_val = range(200, 700)                  # 200,500
    idx_test = range(500, 1500)                # 500,1500s

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
