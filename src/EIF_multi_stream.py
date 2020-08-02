import numpy as np
import sys

from iForest import Node, Copy_Node
from util import *

class iForest_MS(object):
    def __init__(self, dataset, feautre_map, sub_sample = 256):

        self.num_tree = 100

        self.feature_map = feautre_map
        self.dataset = dataset

        self.sigma = [1/4/np.sqrt(self.num_tree) * np.exp(np.random.randn()) for _ in range(len(self.forest))]

        self.forest = []
        self.sub_samples = self.feature_map.sub_sample_sizes
        self.ls = np.ceil(np.log2(self.sub_samples))
        self.bound = []
        # for data in dataset:

        #     if len(data) == 0:
        #         self.sub_samples.append(np.nan)
        #         self.ls.append(np.nan)
        #         self.bound.append(np.array([np.empty([1,data.shape[1]]),np.empty([1,data.shape[1]])], 
        #             dtype=np.float64).T)
        #         continue

            

        for tree_idx, (anchor, sub_sample) in enumerate(zip(self.feature_map.tree_anchors, self.feature_map.sub_sample_sizes)):
            indices = np.where(anchor)[0]
            data = np.concatenate([dataset[idx][:,1:] for idx in indices], axis=0)

            self.bound.append(
                np.array([np.amin(data, axis=0)-1,np.amax(data, axis=0)+1], 
                    dtype=np.float64).T)

            sub_data = data[np.random.choice(len(data), sub_sample, replace=False)]
            bound = self.bound[-1]

            self.forest.append(Node(sub_data, bound, 0, self.ls[tree_idx]))


    def add_trees(self):

        if len(self.ls) != len(self.sub_samples):
            sys.exit('iforest sub sample size is not sychronized with feature map!')
        elif len(self.dataset) != len(self.feature_map.tree_anchors[0]):
            sys.exit('iforest dataset is not sychronized with data holder!')

        else:
            self.sub_samples = self.feature_map.sub_sample_sizes
            self.ls = np.ceil(np.log2(self.sub_samples))
        
        old_len = len(self.forest)
        for i in range(old_len, len(self.feature_map.tree_anchors)):
            anchor = self.feature_map.tree_anchors[i]
            sub_sample = self.feature_map.sub_sample_sizes[i]
            
            indices = np.where(anchor)[0]
            data = np.concatenate([self.dataset[idx][:,1:] for idx in indices], axis=0)

            self.bound.append(
                np.array([np.amin(data, axis=0)-1,np.amax(data, axis=0)+1], 
                    dtype=np.float64).T)

            sub_data = data[np.random.choice(len(data), sub_sample, replace=False)]
            bound = self.bound[-1]

            self.forest.append(Node(sub_data, bound, 0, self.ls[i]))
            self.sigma.append(1/4/np.sqrt(self.num_tree) * np.exp(np.random.randn()))




    # def update_bound(self, sig_idx, sub_sample = 256):
    #     data = self.dataset[sig_idx]

    #     self.sub_samples[sig_idx] = min(sub_sample, len(data))
    #     self.ls[sig_idx] = np.ceil(np.log2(self.sub_samples[sig_idx]))
    #     self.bound[sig_idx] = np.array([np.amin(data, axis=0),np.amax(data, axis=0)], 
    #                 dtype=np.float64).T

    # def insert_trees(self, sig_idx):
    #     insert_indices = np.where(self.feature_map.tree_creation_anchors[:,sig_idx])[0]

    #     if len(insert_indices) == 0:
    #         return

    #     if (insert_indices[-1] < len(self.feature_map.tree_creation_anchors) - 1) |\
    #         (np.max(np.diff(insert_indices)) > 1):
    #         print('something went wrong')
    #         exit()

    #     else:
    #         data = self.dataset[sig_idx]
    #         sub_sample = self.sub_samples[sig_idx]
    #         sub_data = data[np.random.choice(len(data), sub_sample, replace=False)]
    #         bound = self.bound[sig_idx]
    #         for _ in insert_indices:
    #             self.forest.append(Node(sub_data, bound, 0, self.ls[sig_idx]))

    def compute_score(self, pt, signal_idx):

        tree_indices = np.where(self.feature_map.tree_anchors[:,signal_idx] == 1)[0]

        sum_path_len = 0
        ave_ext_l = []
        pll = []
        for idx in tree_indices:
            tree = self.forest[idx]
            avg_ext = tree.avg_external_len(self.sub_samples[idx])
            sum_path_len += tree.path_len(pt,0)/avg_ext
            # print(2**(-tree.path_len(pt,0)/avg_ext), from_onehot(self.feature_map.tree_creation_anchors[idx]), signal_idx)
            # ave_ext_l.append(avg_ext)
            # pll.append(sum_path_len)
        # if 2**(-sum_path_len/len(tree_indices)) > 0.85:
        #     print(tree.left, tree.right)
        # print(self.sub_samples[idx], avg_ext)
        # print('_______')

        return 2**(-sum_path_len/len(tree_indices))


class EIF_MS(iForest_MS):
    def __init__(self, dataset, feautre_map, sub_sample = 256):
        super(EIF_MS, self).__init__(dataset, feautre_map, sub_sample)

        
        # self.sigma = [0 for _ in range(self.num_tree)]

        self.lr = 1/np.sqrt(self.num_tree)

    def update(self, fitness_fuc, num_iter=1):
        pfitness = None
        for iter_count in range(num_iter):
            pfitness, fitness = self.iterate(fitness_fuc)
            # print(pfitness, fitness)
            # if iter_count > 0:
            #     if (fitness - pfitness) < 0.02:
            #         break
            # pfitness = fitness

    def update_tree(self, tree_idx, sig_indices, fitness_fuc, num_iter=1):
        for iter_count in range(num_iter):

            off_springs = [self.forest[tree_idx]]
            off_springs_sigma = [self.sigma[tree_idx]]
            off_size = 5

            for _ in range(off_size):

                if np.random.uniform() < 0.6:

                    parent2_indices = self.feature_map.tree_anchors[:,sig_indices]
                    parent2_indices = np.where(np.where(parent2_indices == 1)[0] != tree_idx)[0]
                    p2_idx = np.random.choice(parent2_indices, 1)[0]

                    child = self.crossover(self.forest[tree_idx], self.forest[p2_idx])
                    child, c_sigma = self.mutate(child, self.sigma[tree_idx], tree_idx)

                else:
                    indices = np.where(self.feature_map.tree_anchors[tree_idx])[0]
                    sig = np.concatenate([self.dataset[idx][:,1:] for idx in indices], axis=0)
                    child = Node(sig[np.random.choice(len(sig), self.sub_samples[tree_idx], replace=False)], self.bound[tree_idx], 0, self.ls[tree_idx])
                    c_sigma = 1/4/np.sqrt(self.num_tree) * np.exp(np.random.randn())

                off_springs.append(child)
                off_springs_sigma.append(c_sigma)

            selected_idx, _, _ = self.selection(off_springs, fitness_fuc, tree_idx)
            # selected_idx = 0

            self.forest[tree_idx] = off_springs[selected_idx]
            self.sigma[tree_idx] = off_springs_sigma[selected_idx]

        return

    # def update_signal(self, sig_idx, fitness_fuc):

    #     tree_indices = np.where(self.feature_map.tree_connection_anchors[:,sig_idx])[0]

    #     trees = [self.forest[idx] for idx in tree_indices]
    #     sigmas = [self.sigma[idx] for idx in tree_indices]
    #     tree_creation_anchors = self.feature_map.tree_creation_anchors[tree_indices]
    #     tree_connection_anchors = self.feature_map.tree_connection_anchors[tree_indices]

    #     pop_size = len(trees)

    #     off_size = int(np.ceil(pop_size*3))
    #     for num in range(off_size):
    #         if np.random.uniform() < 0.8:
    #             parents_idx = np.random.choice(pop_size, 2)
    #             parents = [self.forest[idx] for idx in parents_idx]
    #             sigma = self.sigma[parents_idx[0]]
    #             child = self.crossover(parents[0], parents[1])
    #             tree_creation_anchors = np.append(tree_creation_anchors, 
    #                 [self.feature_map.tree_creation_anchors[parents_idx[0]]], axis=0)
    #             tree_connection_anchors = np.append(tree_connection_anchors, 
    #                 [self.feature_map.tree_connection_anchors[parents_idx[0]]], axis=0)
    #             # child = parents[0]
                
    #             child, c_sigma = self.mutate(child, sigma, sig_idx)
    #         else:
    #             sig = self.dataset[sig_idx]
    #             child = Node(sig[np.random.choice(len(sig), self.sub_samples[sig_idx], replace=False)], self.bound[sig_idx], 0, self.ls[sig_idx])
    #             c_sigma = 1/4/np.sqrt(pop_size) * np.exp(np.random.randn())
    #             creation_anchor = to_onehot(sig_idx, len(tree_creation_anchors[0]))
    #             tree_creation_anchors = np.append(tree_creation_anchors, [creation_anchor], axis=0)
    #             tree_connection_anchors = np.append(tree_connection_anchors, 
    #                 [self.feature_map.crea2conn(creation_anchor)], axis=0)


    #         trees.append(child)
    #         sigmas.append(c_sigma)

    #     selected_idx, p_avg_fitness, avg_fitness = self.selection_group(trees, pop_size, fitness_fuc, sig_idx)

    #     self.forest = [self.forest[idx] for idx in range(len(self.forest)) if not (idx in tree_indices)]
    #     self.sigma = [self.sigma[idx] for idx in range(len(self.sigma)) if not (idx in tree_indices)]

    #     for idx in selected_idx:
    #         self.forest.append(trees[idx])
    #         self.sigma.append(sigmas[idx])

    #     self.feature_map.tree_creation_anchors = np.delete(self.feature_map.tree_creation_anchors, tree_indices, axis=0)
    #     self.feature_map.tree_connection_anchors = np.delete(self.feature_map.tree_connection_anchors, tree_indices, axis=0)
        
    #     self.feature_map.tree_creation_anchors = np.append(self.feature_map.tree_creation_anchors, tree_creation_anchors[selected_idx], axis=0)
    #     self.feature_map.tree_connection_anchors = np.append(self.feature_map.tree_connection_anchors, tree_connection_anchors[selected_idx], axis=0)



    def mutate(self, indi, sig, tree_idx):
        child = Copy_Node(indi)
        sig = sig * np.exp(self.lr*np.random.randn())
        child.mutate(sig, self.bound[tree_idx])

        indices = np.where(self.feature_map.tree_anchors[tree_idx])[0]
        data = np.concatenate([self.dataset[idx][:,1:] for idx in indices], axis=0) 

        child.train(data[np.random.choice(len(data), self.sub_samples[tree_idx], replace=False)])
        return child, sig

    def crossover(self, p1, p2):
        p1c, p2c = Copy_Node(p1), Copy_Node(p2)
        crossover_pt2 = p2c.locate_crossover()
        if crossover_pt2 is not None:
            if_crossed = p1c.set_crossover(crossover_pt2)
            # if if_crossed:
            #     p1c.train(self.data[np.random.choice(len(self.data), self.sub_sample, replace=False)])
        return p1c

    def selection(self, population, fitness_fuc, tree_idx):
        fitness = []
        for indi in population:
            fitness.append(fitness_fuc(indi, tree_idx))

        fitness = np.array(fitness)
        fit_rank = np.argsort(fitness)[::-1]

        selected = fit_rank[0]
        # print('new individual count', np.count_nonzero(selected >= self.num_tree))
        return selected, np.mean(fitness[:self.num_tree]), np.mean(fitness[selected])

    def selection_group(self, population, pop_size, fitness_fuc, sig_idx):
        fitness = []
        for indi in population:
            fitness.append(fitness_fuc(indi, sig_idx))

        fitness = np.array(fitness)
        fit_rank = np.argsort(fitness)[::-1]

        selected = fit_rank[:pop_size]
        # print('new individual count', np.count_nonzero(selected >= self.num_tree))
        return selected, np.mean(fitness[:self.num_tree]), np.mean(fitness[selected])

    # def insert_trees(self, sig_idx):
    #     insert_indices = np.where(self.feature_map.tree_creation_anchors[:,sig_idx])[0]

    #     if len(insert_indices) == 0:
    #         return

    #     if (insert_indices[-1] < len(self.feature_map.tree_creation_anchors) - 1) |\
    #         (np.max(np.diff(insert_indices)) > 1):
    #         print('something went wrong')
    #         exit()

    #     else:
    #         data = self.dataset[sig_idx]
    #         sub_sample = self.sub_samples[sig_idx]
    #         sub_data = data[np.random.choice(len(data), sub_sample, replace=False)]
    #         bound = self.bound[sig_idx]
    #         for _ in insert_indices:
    #             self.forest.append(Node(sub_data, bound, 0, self.ls[sig_idx]))
    #             self.sigma.append(1/4/np.sqrt(self.num_tree) * np.exp(np.random.randn()))

    # def append_trees(self, sig_idx):

    #     len_trees = len(self.forest)
    #     len_anchors = len(self.feature_map.tree_creation_anchors)

    #     for i in range(len_trees, len_anchors):

    #         data = self.dataset[sig_idx]
    #         sub_sample = self.sub_samples[sig_idx]
    #         sub_data = data[np.random.choice(len(data), sub_sample, replace=False)]
    #         bound = self.bound[sig_idx]

    #         self.forest.append(Node(sub_data, bound, 0, self.ls[sig_idx]))
    #         self.sigma.append(1/4/np.sqrt(self.num_tree) * np.exp(np.random.randn()))