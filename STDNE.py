import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import sys
from DataSet import DataSet
from Evaluation import Evaluation

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0


class STDNE:
    def __init__(self, file_path, cl_label_data, emb_size=128, neg_size=10, hist_len=2, directed=False,
                 learning_rate=0.01, batch_size=1000, save_step=50, epoch_num=300, sim_num=256, beta=0.2, walk_len=10,
                 gamma=0.5, t_lambda=0.1):
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num
        self.sim_num = sim_num
        self.walk_len = walk_len
        self.cl_label_data = cl_label_data
        self.beta = beta
        self.gamma = gamma
        self.t_lambda = t_lambda
        self.micro_f1_log = []
        self.macro_f1_log = []

        self.data = DataSet(file_path, neg_size, hist_len, directed, sim_len=sim_num)
        self.time_ind = self.data.get_time_cnt()
        self.G_num = len(self.time_ind)
        self.node_dim = self.data.get_node_dim()

        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                    -1.5 / np.sqrt(self.node_dim), 1.5 / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                    FType).cuda(), requires_grad=True)  # node embedding
                self.time_node_emb = Variable(torch.from_numpy(np.random.uniform(
                    -0.01 / np.sqrt(self.node_dim), 0.01 / np.sqrt(self.node_dim),
                    (self.G_num * self.node_dim, emb_size))).type(
                    FType).cuda(), requires_grad=True)  # temporal embedding

                self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.delta_a = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.delta_b = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.delta_n = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)

                self.att_param = Variable(torch.diag(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(emb_size), 1. / np.sqrt(emb_size), (emb_size,))).type(
                    FType).cuda()), requires_grad=True)
        else:
            self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                FType), requires_grad=True)
            self.time_node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim),
                (self.G_num * self.node_dim, emb_size))).type(
                FType), requires_grad=True)

            self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)

            self.att_param = Variable(torch.diag(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(emb_size), 1. / np.sqrt(emb_size), (emb_size,))).type(
                FType)), requires_grad=True)

        self.opt = SGD(lr=learning_rate,
                       params=[self.node_emb, self.time_node_emb, self.att_param, self.delta, self.delta_a,
                               self.delta_b, self.delta_n])
        self.loss = torch.FloatTensor()

    def struct_forward(self, s_nodes, t_nodes, t_times, t_ind, n_nodes, h_nodes, h_times, h_time_mask, h_time_ind,
                       s_sim,
                       s_sim_time, s_sim_mask, s_sim_ind, t_sim, t_sim_time, t_sim_mask, t_sim_ind, n_sim, n_sim_time,
                       n_sim_mask, n_sim_ind):
        batch = s_nodes.size()[0]

        '''
        Select node embedding, including neighbours and negative sample
        '''
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        a_node_emb = self.node_emb.index_select(0, Variable(s_sim.view(-1))).view(batch, self.sim_num, -1)
        b_node_emb = self.node_emb.index_select(0, Variable(t_sim.view(-1))).view(batch, self.sim_num, -1)
        ns_node_emb = self.node_emb.index_select(0, Variable(n_sim.view(-1))).view(batch, self.neg_size, self.sim_num,
                                                                                   -1)

        '''
        Because we stored time embedding into one big tensor, so it need to transform time index to true index
        for example:
            2nd graph's 345th embedding
            -> index = 2 * nodes + 345
        '''
        s_t_node_ind = t_ind * self.node_dim + s_nodes
        t_t_node_ind = t_ind * self.node_dim + t_nodes
        h_t_node_ind = h_time_ind * self.node_dim + h_nodes
        n_t_node_ind = t_ind.unsqueeze(1) * self.node_dim + n_nodes
        a_t_node_ind = s_sim_ind * self.node_dim + s_sim
        b_t_node_ind = t_sim_ind * self.node_dim + t_sim
        ns_t_node_ind = n_sim_ind * self.node_dim + n_sim

        '''
        Select temporal embedding
        '''
        s_t_node_emb = self.time_node_emb.index_select(0, s_t_node_ind.view(-1)).view(batch, -1)
        t_t_node_emb = self.time_node_emb.index_select(0, t_t_node_ind.view(-1)).view(batch, -1)
        h_t_node_emb = self.time_node_emb.index_select(0, h_t_node_ind.view(-1)).view(batch, self.hist_len, -1)
        n_t_node_emb = self.time_node_emb.index_select(0, n_t_node_ind.view(-1)).view(batch, self.neg_size, -1)
        a_t_node_emb = self.time_node_emb.index_select(0, a_t_node_ind.view(-1)).view(batch, self.sim_num, -1)
        b_t_node_emb = self.time_node_emb.index_select(0, b_t_node_ind.view(-1)).view(batch, self.sim_num, -1)
        ns_t_node_emb = self.time_node_emb.index_select(0, ns_t_node_ind.view(-1)).view(batch, self.neg_size,
                                                                                        self.sim_num, -1)

        '''
        Add time embedding to node embedding
        '''
        s_node_emb = s_node_emb + self.t_lambda * s_t_node_emb
        t_node_emb = t_node_emb + self.t_lambda * t_t_node_emb
        h_node_emb = h_node_emb + self.t_lambda * h_t_node_emb
        n_node_emb = n_node_emb + self.t_lambda * n_t_node_emb
        a_node_emb = a_node_emb + self.t_lambda * a_t_node_emb
        b_node_emb = b_node_emb + self.t_lambda * b_t_node_emb
        ns_node_emb = ns_node_emb + self.t_lambda * ns_t_node_emb

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(dim=1)

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)

        a_alpha = ((a_node_emb - s_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()
        a_delta = self.delta_a.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        a_d_time = torch.abs(t_times.unsqueeze(1) - s_sim_time)
        a_att = softmax(((t_node_emb.unsqueeze(1) - a_node_emb) ** 2).sum(dim=2).neg(), dim=1)
        tmp1 = (a_att * a_alpha * torch.exp(a_delta * Variable(a_d_time)) * Variable(s_sim_mask)).sum(dim=1)
        p_lambda = p_lambda + self.beta * tmp1
        n_lambda = n_lambda + self.beta * tmp1.unsqueeze(1)

        b_alpha = ((b_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()
        b_delta = self.delta_b.index_select(0, Variable(t_nodes.view(-1))).unsqueeze(1)
        b_d_time = torch.abs(t_times.unsqueeze(1) - t_sim_time)
        b_att = softmax(((s_node_emb.unsqueeze(1) - b_node_emb) ** 2).sum(dim=2).neg(), dim=1)
        tmp2 = (b_att * b_alpha * torch.exp(b_delta * Variable(b_d_time)) * Variable(t_sim_mask)).sum(dim=1)
        p_lambda = p_lambda + (1 - self.beta) * tmp2

        ns_alpha = ((ns_node_emb - n_node_emb.unsqueeze(2)) ** 2).sum(dim=3).neg()
        ns_d_time = torch.abs(t_times.unsqueeze(1).unsqueeze(2) - n_sim_time)
        ns_delta = self.delta_a.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        ns_att = softmax(((s_node_emb.unsqueeze(1).unsqueeze(2) - ns_node_emb) ** 2).sum(dim=3).neg(), dim=2)
        tmp3 = (ns_att * ns_alpha * torch.exp(ns_delta.unsqueeze(1) * Variable(ns_d_time)) * Variable(
            n_sim_mask)).sum(dim=2)

        n_lambda = n_lambda + (1.0 - self.beta) * tmp3

        return p_lambda, n_lambda

    def time_forward(self, s_nodes, t_ind, walk_path, walk_neg, walk_mask, walk_neg_mask):
        batch = s_nodes.size()[0]
        s_t_node_ind = t_ind * self.node_dim + s_nodes
        s_time_emb = self.time_node_emb.index_select(0, s_t_node_ind.view(-1)).view(batch, -1)
        s_walk_path = torch.transpose(walk_path, 0, 1).contiguous()
        s_walk_neg = torch.transpose(walk_neg, 0, 1).contiguous()
        s_walk_mask = torch.transpose(walk_mask, 0, 1).contiguous()
        n_walk_mask = torch.transpose(walk_neg_mask, 0, 1).contiguous()
        p_loss = torch.zeros(batch).type(FType).cuda()
        q_loss = torch.zeros(batch).type(FType).cuda()
        for G_ind in range(self.G_num):
            s_walk = s_walk_path[G_ind]
            s_walk_n = s_walk_neg[G_ind]
            s_walk_m = s_walk_mask[G_ind]
            n_walk_m = n_walk_mask[G_ind]
            s_walk_node_ind = G_ind * self.node_dim + s_walk
            n_walk_node_ind = G_ind * self.node_dim + s_walk_n
            s_walk_node_emb = self.time_node_emb.index_select(0, Variable(s_walk_node_ind.view(-1))).view(batch,
                                                                                                          self.walk_len,
                                                                                                          -1)
            n_walk_node_emb = self.time_node_emb.index_select(0, Variable(n_walk_node_ind.view(-1))).view(batch,
                                                                                                          self.neg_size,
                                                                                                          -1)
            # print(s_walk_node_emb.size())
            # print(s_time_emb.size())
            # print(s_walk_m.size())
            p = torch.mul(torch.mul(s_time_emb.unsqueeze(1), s_walk_node_emb), s_walk_m.unsqueeze(2))
            q = torch.mul(torch.mul(s_time_emb.unsqueeze(1), n_walk_node_emb), n_walk_m.unsqueeze(2))
            p_loss += p.sum(2).sum(1)
            q_loss += q.sum(2).sum(1)
        return p_loss, q_loss

    def struct_loss(self, s_nodes, t_nodes, t_times, t_ind, n_nodes, h_nodes, h_times, h_time_mask, h_time_ind, s_sim,
                    s_sim_time, s_sim_mask, s_sim_ind, t_sim, t_sim_time, t_sim_mask, t_sim_ind, n_sim, n_sim_time,
                    n_sim_mask, n_sim_ind):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                p_lambdas, n_lambdas = self.struct_forward(s_nodes, t_nodes, t_times, t_ind, n_nodes, h_nodes, h_times,
                                                           h_time_mask, h_time_ind, s_sim,
                                                           s_sim_time, s_sim_mask, s_sim_ind, t_sim, t_sim_time,
                                                           t_sim_mask,
                                                           t_sim_ind, n_sim, n_sim_time,
                                                           n_sim_mask, n_sim_ind)
                loss = -torch.log(p_lambdas.sigmoid() + 1e-6) - torch.log(
                    n_lambdas.neg().sigmoid() + 1e-6).sum(dim=1)

        else:
            p_lambdas, n_lambdas = self.struct_forward(s_nodes, t_nodes, t_times, t_ind, n_nodes, h_nodes, h_times,
                                                       h_time_mask, h_time_ind, s_sim,
                                                       s_sim_time, s_sim_mask, s_sim_ind, t_sim, t_sim_time, t_sim_mask,
                                                       t_sim_ind, n_sim, n_sim_time,
                                                       n_sim_mask, n_sim_ind)
            loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) - torch.log(
                torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)
        return loss

    def time_loss(self, s_nodes, t_ind, walk_path, walk_neg, walk_mask, walk_neg_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                p_, q_ = self.time_forward(s_nodes, t_ind, walk_path, walk_neg, walk_mask, walk_neg_mask)
                loss = -torch.log(p_.sigmoid() + 1e-6) - torch.log(q_.neg().sigmoid() + 1e-6)

        else:
            p_, q_ = self.time_forward(s_nodes, t_ind, walk_path, walk_neg, walk_mask, walk_neg_mask)
            loss = -torch.log(p_.sigmoid() + 1e-6) - torch.log(q_.neg().sigmoid() + 1e-6)
        return loss

    def update(self, s_nodes, t_nodes, t_times, t_ind, n_nodes, h_nodes, h_times, h_time_mask, h_time_ind, s_sim,
               s_sim_time, s_sim_mask, s_sim_ind, t_sim, t_sim_time, t_sim_mask, t_sim_ind, n_sim, n_sim_time,
               n_sim_mask, n_sim_ind, walk_path, walk_neg, walk_mask, walk_neg_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                t_loss = self.time_loss(s_nodes, t_ind, walk_path, walk_neg, walk_mask, walk_neg_mask)
                s_loss = self.struct_loss(s_nodes, t_nodes, t_times, t_ind, n_nodes, h_nodes, h_times, h_time_mask,
                                          h_time_ind, s_sim,
                                          s_sim_time, s_sim_mask, s_sim_ind, t_sim, t_sim_time, t_sim_mask, t_sim_ind,
                                          n_sim, n_sim_time,
                                          n_sim_mask, n_sim_ind)

                loss = self.gamma * s_loss.sum() + (1 - self.gamma) * t_loss.sum()
                self.loss += loss.data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            t_loss = self.time_loss(s_nodes, t_ind, walk_path, walk_neg, walk_mask, walk_neg_mask)
            s_loss = self.struct_loss(s_nodes, t_nodes, t_times, t_ind, n_nodes, h_nodes, h_times, h_time_mask,
                                      h_time_ind,
                                      s_sim,
                                      s_sim_time, s_sim_mask, s_sim_ind, t_sim, t_sim_time, t_sim_mask, t_sim_ind,
                                      n_sim,
                                      n_sim_time,
                                      n_sim_mask, n_sim_ind)
            loss = self.gamma * s_loss.sum() + (1 - self.gamma) * t_loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch,
                                shuffle=True, num_workers=10)
            if epoch % self.save_step == 0 and epoch != 0:
                self.save_node_embeddings('./emb/dblp_%d.emb' % (epoch))

            if epoch % 1 == 0 and epoch != 0:
                print('evaluation...')
                if torch.cuda.is_available():
                    emb = self.node_emb.clone().detach()
                    t_emb = torch.mean(torch.stack(torch.chunk(self.time_node_emb, self.G_num, 0)), dim=0)
                    emb += self.t_lambda * t_emb
                    # embeddings = self.node_emb.cpu().data.numpy()
                    embeddings = emb.cpu().data.numpy()
                else:
                    emb = self.node_emb.clone().detach()
                    t_emb = torch.mean(torch.stack(torch.chunk(self.time_node_emb, self.G_num, 0)), dim=0)
                    emb += self.t_lambda * t_emb
                    embeddings = emb.data.numpy()
                    # embeddings = self.node_emb.data.numpy()
                eva = Evaluation(emb_data=embeddings, from_file=False)
                mi, ma = eva.lr_classification(train_ratio=0.8, label_data=self.cl_label_data)
                self.micro_f1_log.append(mi)
                self.macro_f1_log.append(ma)

            if epoch % 50 == 0 and epoch:
                print('Epoch {}\nBest Micro F1 Score: {}\nBest Macro F1 Score: {}'.format(epoch, max(self.micro_f1_log),
                                                                                          max(self.macro_f1_log)))

            for i_batch, sample_batched in enumerate(loader):
                if i_batch % 100 == 0 and i_batch != 0:
                    print(str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)) + '\tdelta:' + str(
                        self.delta.mean().cpu().data.numpy()))

                if torch.cuda.is_available():
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['target_time'].type(FType).cuda(),
                                    sample_batched['target_time_ind'].type(LType).cuda(),
                                    sample_batched['neg_nodes'].type(LType).cuda(),
                                    sample_batched['history_nodes'].type(LType).cuda(),
                                    sample_batched['history_times'].type(FType).cuda(),
                                    sample_batched['history_masks'].type(FType).cuda(),
                                    sample_batched['history_ind'].type(LType).cuda(),
                                    sample_batched['s_sim'].type(LType).cuda(),
                                    sample_batched['s_sim_time'].type(FType).cuda(),
                                    sample_batched['s_sim_mask'].type(FType).cuda(),
                                    sample_batched['s_sim_ind'].type(LType).cuda(),
                                    sample_batched['t_sim'].type(LType).cuda(),
                                    sample_batched['t_sim_time'].type(FType).cuda(),
                                    sample_batched['t_sim_mask'].type(FType).cuda(),
                                    sample_batched['t_sim_ind'].type(LType).cuda(),
                                    sample_batched['n_sim'].type(LType).cuda(),
                                    sample_batched['n_sim_time'].type(FType).cuda(),
                                    sample_batched['n_sim_mask'].type(FType).cuda(),
                                    sample_batched['n_sim_ind'].type(LType).cuda(),
                                    sample_batched['walk_path'].type(LType).cuda(),
                                    sample_batched['walk_neg'].type(LType).cuda(),
                                    sample_batched['walk_mask'].type(LType).cuda(),
                                    sample_batched['neg_walk_mask'].type(LType).cuda()
                                    )
                        # print(i_batch)
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['target_time_ind'].type(LType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType),
                                sample_batched['s_sim'].type(LType).cuda(),
                                sample_batched['s_sim_time'].type(FType).cuda(),
                                sample_batched['s_sim_mask'].type(FType).cuda(),
                                sample_batched['t_sim'].type(LType).cuda(),
                                sample_batched['t_sim_time'].type(FType).cuda(),
                                sample_batched['t_sim_mask'].type(FType).cuda(),
                                sample_batched['n_sim'].type(LType).cuda(),
                                sample_batched['n_sim_time'].type(FType).cuda(),
                                sample_batched['n_sim_mask'].type(FType).cuda())

            print('epoch ' + str(epoch) + ': avg loss = ' + str(self.loss.cpu().numpy() / len(self.data)) + '\n')

        self.save_node_embeddings('./emb/dblp_%d.emb' % self.epochs)
        print('Training finished\nBest Micro F1 Score: {}\nBest Macro F1 Score: {}'.format(max(self.micro_f1_log),
                                                                                           max(self.macro_f1_log)))

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()


if __name__ == '__main__':
    m = STDNE('./data/dblp/dblp.txt', './data/dblp/node2label.txt', directed=False)
    m.train()
