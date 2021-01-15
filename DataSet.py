from torch.utils.data import Dataset
import numpy as np
import sys
import random
import networkx as nx
import matplotlib.pyplot as plt
import os


class DataSet(Dataset):
    def __init__(self, file_path, neg_size, hist_len, directed=False, transform=None, walk_len=10, sim_len=10):
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.directed = directed
        self.transform = transform
        self.walk_len = walk_len

        self.max_d_time = -sys.maxsize  # Time interval [0, T]

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)

        self.node2hist = dict()
        self.node_set = set()
        self.degrees = dict()
        self.G_list = {}
        self.time_ind = []
        self.sim_len = sim_len
        with open(file_path, 'r') as infile:
            for line in infile:
                parts = line.split()
                s_node = int(parts[0])  # source node
                t_node = int(parts[1])  # target node
                d_time = float(parts[2])  # time slot, delta t
                if d_time not in self.G_list:
                    self.G_list.update({d_time: nx.Graph()})
                    self.time_ind.append(d_time)

                self.node_set.update([s_node, t_node])
                self.G_list[d_time].add_edge(s_node, t_node)

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, d_time, self.time_ind.index(d_time)))

                if not directed:
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, d_time, self.time_ind.index(d_time)))

                if d_time > self.max_d_time:
                    self.max_d_time = d_time

                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0
                self.degrees[s_node] += 1
                self.degrees[t_node] += 1

        self.node_dim = len(self.node_set)

        self.G_num = len(self.G_list)
        print('Generate {} temporal graphs'.format(self.G_num))
        self.time_ind.sort()

        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()

        self.gdv_list = {}
        for i in range(len(self.time_ind)):
            with open('convert/dblp/{}.out'.format(i)) as file:
                a = file.readlines()
                vec = np.ndarray((len(a), 73))
                leg = []
                for j in range(len(a)):
                    vec[j] = [int(k) for k in a[j].split(' ')]
                    if np.count_nonzero(vec[j]) > 30:
                        leg.append(j)
            self.gdv_list.update({self.time_ind[i]: {'array': vec, 'legal': leg}})

        self.walk_path = {}
        self.init_walk_path()
        self.gdv_sim = {}
        self.init_gdv_table()

    def get_node_dim(self):
        return self.node_dim

    def get_max_d_time(self):
        return self.max_d_time

    def get_time_cnt(self):
        return self.time_ind

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        for k in range(self.node_dim):
            tot_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def init_gdv_table(self):
        if os.path.exists('gdv_t.npy'):
            print('loading gdv table')
            self.gdv_sim = np.load('gdv_t.npy', allow_pickle=True)
            self.gdv_sim = self.gdv_sim.item()
            print('finished')
            return
        print('generating gdv table')
        for t in self.time_ind:
            print(t)
            array = self.gdv_list[t]['array']
            self.gdv_sim.update({t: {}})
            leg = self.gdv_list[t]['legal']
            for pt in range(array.shape[0]):
                if pt not in leg:
                    self.gdv_sim[t].update({pt: None})
                    continue
                else:
                    gdv = array[pt]
                    self.gdv_sim[t].update({pt: []})
                    b = np.reshape(gdv, (1, -1))
                    for tt in self.time_ind:
                        if tt == t:
                            continue
                        a = self.gdv_list[tt]['array']
                        a_leg = self.gdv_list[tt]['legal']
                        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
                        b_norm = np.linalg.norm(b)
                        similiarity = np.dot(a, b.T) / ((a_norm + 1e-8) * b_norm)
                        dist = 1. - similiarity
                        p = dist.reshape((-1)).argsort()
                        for k in range(10):
                            if int(p[k]) in a_leg:
                                self.gdv_sim[t][pt].append((p[k], tt, self.time_ind.index(tt)))
        np.save('gdv_t.npy', self.gdv_sim, allow_pickle=True)

    def init_walk_path(self):
        if os.path.exists('walk_p.npy'):
            print('loading walk path')
            self.walk_path = np.load('walk_p.npy', allow_pickle=True)
            self.walk_path = self.walk_path.item()
            print('finished')
            return
        print('generating walk path')
        for t in self.time_ind:
            print(t)
            self.walk_path.update({t: {}})
            G = self.G_list[t]
            for node in self.node_set:
                try:
                    allp = []
                    for ite in range(30):
                        _n = node
                        rdw = []
                        _vis = [_n]
                        for i in range(20):
                            nei = list(filter(lambda k: k not in _vis, list(G.neighbors(_n))))
                            if nei:
                                _n = random.choice(nei)
                                _vis.append(_n)
                                rdw.append(_n)
                            else:
                                break
                        mrd = max(self.node_set)
                        _neg = []
                        for i in range(40):
                            rd = random.randint(0, mrd)
                            if rd not in rdw and rd not in _neg:
                                _neg.append(rd)
                        allp.append((rdw, _neg))
                    self.walk_path[t].update({node: allp})
                except nx.exception.NetworkXError as e:
                    self.walk_path[t].update({node: None})
        np.save('walk_p.npy', self.walk_path, allow_pickle=True)

    def random_walk(self, node, time):
        np_walk_path = np.zeros((len(self.G_list), self.walk_len))
        np_walk_neg = np.zeros((len(self.G_list), self.neg_size))
        np_walk_mask = np.zeros((len(self.G_list), self.walk_len))
        np_walk_neg_mask = np.zeros((len(self.G_list), self.neg_size))
        G_ind = 0
        for tt in self.time_ind:
            if tt < time:
                if self.walk_path[tt][node]:
                    ch = random.choice(self.walk_path[tt][node])
                    pth = ch[0].copy()
                    neg = ch[1].copy()
                    for i in range(min(len(pth), self.walk_len)):
                        np_walk_path[G_ind, i] = pth[i]
                        np_walk_mask[G_ind, i] = 1
                    for i in range(self.neg_size):
                        np_walk_neg[G_ind, i] = neg[i]
                        np_walk_neg_mask[G_ind, i] = 1
                else:
                    pass
            G_ind += 1
        return np_walk_path, np_walk_neg, np_walk_mask, np_walk_neg_mask

    def find_sim(self, vec_ind, vec_time):
        ls = self.gdv_sim[vec_time][vec_ind]
        res = np.zeros(self.sim_len)
        tt = np.zeros(self.sim_len)
        tt_mask = np.zeros(self.sim_len)
        tt_ind = np.zeros(self.sim_len)
        if ls is None:
            return res, tt, tt_mask, tt_ind
        random.shuffle(ls)
        for i in range(min(len(ls), self.sim_len)):
            res[i] = ls[i][0]
            tt[i] = ls[i][1]
            tt_ind[i] = ls[i][2]
            tt_mask[i] = 1
        return res, tt, tt_mask, tt_ind

    def find_neg_sim(self, vec_ind, vec_time):
        np_res = np.zeros((self.neg_size, self.sim_len))
        np_tm = np.zeros((self.neg_size, self.sim_len))
        np_tm_mask = np.zeros((self.neg_size, self.sim_len))
        np_tm_ind = np.zeros((self.neg_size, self.sim_len))
        i = -1
        for ind in vec_ind:
            i += 1
            if ind not in self.gdv_list[vec_time]['legal']:
                continue
            ls = self.gdv_sim[vec_time][ind]
            random.shuffle(ls)
            for j in range(min(len(ls), self.sim_len)):
                np_res[i, j] = ls[j][0]
                np_tm[i, j] = ls[j][1]
                np_tm_ind[i, j] = ls[j][2]
                np_tm_mask[i, j] = 1
        return np_res, np_tm, np_tm_mask, np_tm_ind

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]
        t_time_ind = self.node2hist[s_node][t_idx][2]

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]
        hist_ind = [self.time_ind.index(h[1]) for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.
        np_h_ind = np.zeros((self.hist_len,))
        np_h_ind[:len(hist_times)] = hist_ind

        s_sim, s_sim_time, s_sim_mask, s_sim_ind = self.find_sim(s_node, t_time)
        t_sim, t_sim_time, t_sim_mask, t_sim_ind = self.find_sim(t_node, t_time)

        neg_nodes = self.negative_sampling()
        n_sim, n_sim_time, n_sim_mask, n_sim_time_ind = self.find_neg_sim(neg_nodes, t_time)

        np_walk_path, np_walk_neg, np_walk_mask, np_walk_neg_mask = self.random_walk(s_node, t_time)

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'target_time_ind': t_time_ind,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_masks': np_h_masks,
            'history_ind': np_h_ind,
            'neg_nodes': neg_nodes,
            'walk_path': np_walk_path,
            'walk_neg': np_walk_neg,
            'walk_mask': np_walk_mask,
            'neg_walk_mask': np_walk_neg_mask,
            's_sim': s_sim,
            's_sim_time': s_sim_time,
            's_sim_mask': s_sim_mask,
            's_sim_ind': s_sim_ind,
            't_sim': t_sim,
            't_sim_time': t_sim_time,
            't_sim_mask': t_sim_mask,
            't_sim_ind': t_sim_ind,
            'n_sim': n_sim,
            'n_sim_time': n_sim_time,
            'n_sim_mask': n_sim_mask,
            'n_sim_ind': n_sim_time_ind,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
        sampled_nodes = self.neg_table[rand_idx]
        return sampled_nodes


if __name__ == '__main__':
    a = DataSet('./data/dblp/dblp.txt', 10, 10, False)
    print(a[213]['s_sim'])
    print(a[213]['s_sim_time'])
