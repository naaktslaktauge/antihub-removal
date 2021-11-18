import faiss
import numpy as np


class AntihubRemover:
    def __init__(self, k, d, gpu_list=[], size_subset=10000000):
        '''
        [INPUT]
        - k [integer]: The only hyperparameter of anti-hub removal.
            This specifies the number of vectors 
            which are considered as NNs in the calculation of hubness.
            Ex. 16
        - d [integer]: The dimensionality of input vectors.
            Ex. 128
        - gpu_list [Array<integer>]: The list of GPUs you use in the calculation of hubness.
            Ex. [0, 1, 2, 3]
        - size_subset [integer, default 10,000,000]: The size of subset in query search.
            If you run out of the memory of your PC, set the smaller value such as 1,000,000.
        '''
        self.k = k
        self.d = d
        self.gpu_list = gpu_list
        self.size_subset = size_subset
        if gpu_list == []:
            self._construct_cpu_index()
        else:
            self._construct_gpu_index()

    def _construct_cpu_index(self):
        self.index = faiss.IndexFlatL2(self.d)

    def _construct_gpu_index(self):
        resources = [faiss.StandardGpuResources() for _ in self.gpu_list]
        configs = []
        for gpu in self.gpu_list:
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = gpu
            configs.append(cfg)

        indexes = [faiss.GpuIndexFlatL2(resources[i], self.d, configs[i])
                    for i in range(len(self.gpu_list))]
        self.index = faiss.IndexReplicas()
        for index in indexes:
            self.index.addIndex(index)

    def _search(self, index, xq, k, size_subset):
        n_subset = xq.shape[0] // size_subset
        if xq.shape[0] % size_subset > 0:
            n_subset += 1
        D, I = [], []
        for i in range(n_subset):
            D_, I_ = index.search(np.ascontiguousarray(xq[i*size_subset:(i+1)*size_subset]), k)
            D.append(D_)
            I.append(I_)
        D = np.concatenate(D)
        I = np.concatenate(I)
        return D, I
    
    def _clustering(self, xt, xb, n_cluster):
        kmeans = faiss.Kmeans(k=n_cluster, niter=60, d=self.d)
        kmeans.train(np.ascontiguousarray(xt))
        self.index.reset()
        self.index.add(kmeans.centroids)
        _, clusters = self._search(self.index, xb, 1, self.size_subset)
        clusters = clusters.flatten()
        return clusters

    def calc_hubness(self, xb):
        '''
        Calculation of the hubness.
        [Input]
        - xb [np.array(float32)]: Database vectors to remove its anti-hubs.
        [OUTPUT]
        - hubness [Array<integer>]: The hubness score of each vector.
        '''
        self.index.reset()
        self.index.add(np.ascontiguousarray(xb))
        _, I = self._search(self.index, xb, self.k+1, self.size_subset)
        I = I[:, 1:].flatten()
        hubness = [0] * xb.shape[0]
        for i in I:
            hubness[i] += 1
        return hubness
        
    def remove_antihub(self, xb, alpha=0.5, return_vecs=False):
        '''
        Pure anti-hub removal.
        After the calculation of the hubness, low hubness vectors get removed.
        [Input]
        - xb [np.array(float32)]: Database vectors to remove its anti-hubs.
        - alpha [float, 0 < alpha < 1, default 0.5]: The size of database after the anti-hub removal (xb').
            When you set alpha to 0.5, half of vectors are removed from the original database.
            |xb'| = alpha * |xb|
        - return_vecs [bool, default False]: 
            If True, it returns 1) the reduced database and 2) the IDs of remained vectors.
            If False, only IDs is returned.
        
        [OUTPUT]
        - hub_id [np.array<int64>]: The IDs of remained vectors.
        - reduced_xb [np.array<float64>]: The reduced database.
        '''
        if alpha == 1:
            if return_vecs:
                return xb, np.arange(xb.shape[0])
            else:
                return np.arange(xb.shape[0])
        elif 0 < alpha < 1:
            size_xb = int(xb.shape[0] * alpha)
        else:
            AssertionError
        hubness = self.calc_hubness(xb)
        hub_id = np.argsort(hubness)[::-1][:size_xb]
        if return_vecs:
            reduced_xb = xb[hub_id]
            return reduced_xb, hub_id
        else:
            return hub_id

    def remove_approximated_antihub(self, xt, xb, alpha=0.5, n_cluster=1000, return_vecs=False):
        '''
        Approximate anti-hub removal.
        Firstly, kmeans clustering is executed on the database.
        In each cluster,  the hubness is calculated and low hubness vectors get removed.
        [Input]
        - xt [np.array(float32)]: Training vectors which are used in kmeans clustering.
        - xb [np.array(float32)]: Database vectors to remove its anti-hubs.
        - alpha [float, 0 < alpha < 1, default 0.5]: The size of database after the anti-hub removal (xb').
            When you set alpha to 0.5, half of vectors are removed from the original database.
            |xb'| = alpha * |xb|
        - n_cluster [int]: The number of clusters. 
            The more clusters you use, the faster you can calculate hubness.
        - return_vecs [bool, default False]: 
            If True, it returns 1) the reduced database and 2) the IDs of remained vectors.
            If False, only IDs is returned.
        
        [OUTPUT]
        - hub_id [np.array<int64>]: The IDs of remained vectors.
        - reduced_xb [np.array<float64>]: The reduced database.
        '''
        clusters = self._clustering(xt, xb, n_cluster)
        hub_ids = []
        for cluster in range(n_cluster):
            hub_id = self.remove_antihub(xb[clusters==cluster], alpha)
            hub_id = np.arange(xb.shape[0])[clusters==cluster][hub_id]
            hub_ids.append(hub_id)
        hub_id = np.concatenate(hub_ids)
        if return_vecs:
            reduced_xb = xb[hub_id]
            return reduced_xb, hub_id
        else:
            return hub_id
