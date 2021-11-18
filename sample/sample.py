import os
import sys

import numpy as np
import faiss

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.antihub_remover import AntihubRemover

def validate(index, xb_small, id_small, xq, gt):
    # execute search and calculate recall@1
    index.reset()
    index.add(xb_small)
    _, I = index.search(xq, 1)
    I = id_small[I]
    recall_1 = (I[:, :1] == gt[:, :1]).sum() / gt.shape[0]
    return recall_1

# fix seed
np.random.seed(42)

# make dataset (database vectors and queries)
N, Nq, d = 100000, 1000, 32
X = np.random.random((N, d)).astype(np.float32)  # database: 10,000 32-dim vectors
Xq = np.random.random((Nq, d)).astype(np.float32)  # query: 1,000 32-dim vectors

# make ground truth for each query
index = faiss.IndexFlatL2(d)
index.add(X)
_, gt = index.search(Xq, 1)

# antihub-removal and random reduction
print('Original Data Size: ({}, {})'.format(X.shape[0], X.shape[1]))
ahr = AntihubRemover(k=16, d=d)
# pure anti-hub removal
X_small, hub_id = ahr.remove_antihub(X, alpha=0.5, return_vecs=True)
# approximate anti-hub removal
X_small_appr, hub_id_appr = ahr.remove_approximated_antihub(X[:10000], X, alpha=0.5, n_cluster=10, return_vecs=True)
print('Reduced Data Size: ({}, {})'.format(X_small.shape[0], X_small.shape[1]))
# random reduction
rand_id = np.random.choice(np.arange(N), X_small.shape[0], replace=False)
X_small_rand = X[rand_id]

# validation
recall_1 = validate(index, X_small, hub_id, Xq, gt)
recall_1_appr = validate(index, X_small_appr, hub_id_appr, Xq, gt)
recall_1_rand = validate(index, X_small_rand, rand_id, Xq, gt)
print('recall@1 (random reduction) {:.2f}'.format(recall_1_rand))
print('recall@1 (pure anti-hub removal) {:.2f}'.format(recall_1))
print('recall@1 (approximate anti-hub removal) {:.2f}'.format(recall_1_appr))