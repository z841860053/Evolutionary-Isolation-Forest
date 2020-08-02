import numpy as np

class Empty_Node(object):

    def __init__(self):
        self.left = None
        self.right = None
        self.size = -1
        self.p = None
        self.attribute, self.split = None, None

    def path_len(self, pt, e = 0):
        if self.size != -1:
            return e + self.avg_external_len(self.size)
        if pt[self.attribute] < self.split:
            return self.left.path_len(pt, e+1)
        else:
            return self.right.path_len(pt, e+1)

    def avg_external_len(self, size):
        if (self.size == 1) | (self.size == 0):
            return 0       
        return 2*np.log(size-1) + 0.5772156649 - (2*(size-1)/size)

    def print_split(self):
        if self.split != None:
            print(self.attribute, self.split)
            self.left.print_split()
            self.right.print_split()

class Node(Empty_Node):

    def __init__(self, data, bound, e, l):
        super(Node, self).__init__()
        # print(bound, np.max(data, axis=0), np.min(data, axis=0))
        self.train(data, bound, e, l)

    def train(self, data, bound, e, l):
        if (e >= l) | (len(data) <= 1):
            self.size = len(data)
        else:
            self.attribute = np.random.randint(len(data[0]))
            self.split = np.random.uniform(bound[self.attribute,0], bound[self.attribute,1])
            idx_l = (data[:, self.attribute] < self.split)
            data_l = data[np.where(idx_l)[0]]
            data_r = data[np.where(1-idx_l)[0]]
            self.p = len(data_l)/(len(data_l)+len(data_r))
            self.left = Node(data_l, bound, e+1, l)
            self.right = Node(data_r, bound, e+1, l)

class Traced_Node(Empty_Node):
    def __init__(self, data, bound, e, l):
        super(Traced_Node, self).__init__()
        self.train(data, bound, e, l)

    def train(self, data, bound, e, l):
        if (e >= l) | (len(data) <= 1):
            self.size = len(data)
        else:
            self.attribute = np.random.randint(len(data[0]))
            self.split = np.random.uniform(bound[self.attribute,0], bound[self.attribute,1])
            idx_l = (data[:, self.attribute] < self.split)
            data_l = data[np.where(idx_l)[0]]
            data_r = data[np.where(1-idx_l)[0]]
            self.p = len(data_l)/(len(data_l)+len(data_r))
            self.left = Traced_Node(data_l, bound, e+1, l)
            self.right = Traced_Node(data_r, bound, e+1, l)

    def retrain(self, data):
        if self.size != -1:
            self.size = len(data)

        else:
            idx_l = (data[:, self.attribute] < self.split)
            data_l = data[np.where(idx_l)[0]]
            data_r = data[np.where(1-idx_l)[0]]
            self.left.retrain(data_l)
            self.right.retrain(data_r)
    
    def path_len(self, pt, e = 0):
        if self.size != -1:
            return e + self.avg_external_len(self.size), id(self)
        if pt[self.attribute] < self.split:
            return self.left.path_len(pt, e+1)
        else:
            return self.right.path_len(pt, e+1)


class Weighted_Node(Empty_Node):

    def __init__(self, data, bound, e, l):
        super(Weighted_Node, self).__init__()
        self.train(data, bound, e, l)
        self.w = np.random.uniform()

    def path_len(self, pt, e = 0):
        if self.size != -1:
            return (e + self.avg_external_len(self.size))*self.w
        if pt[self.attribute] < self.split:
            return self.left.path_len(pt, e+1)
        else:
            return self.right.path_len(pt, e+1)

    def update_weight(self, pt, amount, e = 0):
        if self.size != -1:
            self.w += amount * (e + self.avg_external_len(self.size))
            self.w = max(min(self.w, 1),0)
        elif pt[self.attribute] < self.split:
            self.left.update_weight(pt, amount, e+1)
        else:
            self.right.update_weight(pt, amount, e+1)

    def train(self, data, bound, e, l):
        if (e >= l) | (len(data) <= 1):
            self.size = len(data)
        else:
            self.attribute = np.random.randint(len(data[0]))
            self.split = np.random.uniform(bound[self.attribute,0], bound[self.attribute,1])
            idx_l = (data[:, self.attribute] < self.split)
            data_l = data[np.where(idx_l)[0]]
            data_r = data[np.where(1-idx_l)[0]]
            self.p = len(data_l)/(len(data_l)+len(data_r))
            self.left = Weighted_Node(data_l, bound, e+1, l)
            self.right = Weighted_Node(data_r, bound, e+1, l)

class Copy_Node(Empty_Node):
    """docstring for Copy_Node"""
    def __init__(self, node):
        super(Copy_Node, self).__init__()
        self.size = node.size
        self.w = 0.5
        
        self.p = node.p
        self.attribute = node.attribute
        self.split = node.split
        if node.left is not None:
            self.left = Copy_Node(node.left)
        if node.right is not None:
            self.right = Copy_Node(node.right)

    def path_len(self, pt, e = 0):
        if self.size != -1:
            return (e + self.avg_external_len(self.size))*self.w
        if pt[self.attribute] < self.split:
            return self.left.path_len(pt, e+1)
        else:
            return self.right.path_len(pt, e+1)

    def train(self, data):
        if self.split is None:
            self.size = len(data)
        else:
            idx_l = (data[:, self.attribute] < self.split)
            data_l = data[np.where(idx_l)[0]]
            data_r = data[np.where(1-idx_l)[0]]
            # self.p = len(data_l)/(len(data_l)+len(data_r))
            # if self.left is not None:
            self.left.train(data_l)
            # if self.right is not None:
            self.right.train(data_r)

    def mutate(self, sigma, bound):
        if self.split is not None:
            if np.random.uniform() < sigma:
                self.attribute = np.random.randint(len(bound))
                self.split = np.random.uniform(bound[self.attribute,0], bound[self.attribute,1])
            self.split = self.split + sigma * np.random.randn() * (bound[self.attribute,1] - bound[self.attribute,0])
            self.split = min(max(self.split, bound[self.attribute,0]), bound[self.attribute,1])

            self.left.mutate(sigma, bound)
            self.right.mutate(sigma, bound)
        else:
            return
            # self.w = min(max(self.w + sigma * np.random.randn(), 0), 1)



    def locate_crossover(self, p = 0.3):
        rand = np.random.uniform()
        if self.split is None:
            return None
        if rand < p/2:
            return self.left
        elif rand < p:
            return self.right
        elif rand < p + (1-p)/2:
            return self.left.locate_crossover(p)
        else:
            return self.right.locate_crossover(p)

    def set_crossover(self, node, p = 0.3):
        rand = np.random.uniform()
        if self.split is None:
            return False
        if rand < p/2:
            self.left = node
            return True
        elif rand < p:
            self.right = node
            return True
        elif rand < p + (1-p)/2:
            return self.left.set_crossover(node, p)
        else:
            return self.right.set_crossover(node, p)

class iForest(object):
    def __init__(self, data, bound, num_tree, sub_sample = 256):
        self.forest = []
        if len(data) < sub_sample:
            sub_sample = len(data)
        self.l = np.ceil(np.log2(sub_sample))
        self.sub_sample = sub_sample
        self.num_tree = num_tree
        self.bound = bound
        for _ in range(num_tree):
            sub_data = data[np.random.choice(len(data), sub_sample, replace=False)]
            self.forest.append(Node(sub_data, bound, 0, self.l))

    def compute_score(self, pt):
        sum_path_len = 0
        path_len = []
        for tree in self.forest:
            sum_path_len += tree.path_len(pt,0)
            path_len.append(tree.path_len(pt,0))
        path_len = np.array(path_len)
        return 2**(-sum_path_len/self.num_tree/self.forest[0].avg_external_len(self.sub_sample)), 2**(-path_len/self.forest[0].avg_external_len(self.sub_sample))


    def predict(self, data, length):
        rand_sample = data
        score_lst = []
        for x in rand_sample:
            score_lst.append(self.compute_score(x)[0])
        score_lst = np.array(score_lst)
        x = np.random.choice(np.argsort(score_lst), length, p=self.exp_decrease(len(score_lst), 0.1))
        # y = rand_sample[np.argsort(score_lst)][np.random.choice(length, length)][:,0]
        # print(rand_sample[np.argsort(score_lst)])
        # return y
        return rand_sample[x]
        # print(self.roulette_selection(score_lst, length))
        # return np.mean(rand_sample[self.roulette_selection(score_lst, length)], axis=0)


    def exp_decrease(self, length, alpha):
        x = np.ones(length)*alpha
        for i in range(1, length):
            x[i:] = x[i:] * (1-alpha)
        return x/np.sum(x)

class iForest_ES_wrapper(iForest):
    def __init__(self, data, bound, num_tree, sub_sample = 256):
        super(iForest_ES_wrapper, self).__init__(data, bound, num_tree, sub_sample)
        # self.individuals = self.forest
        self.sigma = [1/4/np.sqrt(self.num_tree) * np.exp(np.random.randn()) for _ in range(self.num_tree)]
        # self.sigma = [0 for _ in range(self.num_tree)]
        self.data = data
        self.bound = bound
        self.lr = 1/np.sqrt(self.num_tree)
        self.greedy = True
        # self.lr = 0


    def update(self, fitness_fuc, num_iter=1):
        pfitness = None
        for iter_count in range(num_iter):
            pfitness, fitness = self.iterate(fitness_fuc)
            # print(pfitness, fitness)
            # if iter_count > 0:
            #     if (fitness - pfitness) < 0.02:
            #         break
            # pfitness = fitness


    def iterate(self, fitness_fuc):
        off_size = self.num_tree*3
        for num in range(off_size):
            if np.random.uniform() < 0.8:

                parents_idx = np.random.choice(self.num_tree, 2)
                parents = [self.forest[idx] for idx in parents_idx]
                sigma = self.sigma[parents_idx[0]]
                child = self.crossover(parents[0], parents[1])
                # child = parents[0]
                
                child, c_sigma = self.mutate(child, sigma)
            else:
                child = Node(self.data[np.random.choice(len(self.data), self.sub_sample, replace=False)], self.bound, 0, self.l)
                c_sigma = 1/4/np.sqrt(self.num_tree) * np.exp(np.random.randn())
            self.forest.append(child)
            self.sigma.append(c_sigma)

        selected_idx, p_avg_fitness, avg_fitness = self.selection(self.forest, fitness_fuc)
        self.forest = [self.forest[idx] for idx in selected_idx]
        self.sigma = [self.sigma[idx] for idx in selected_idx]

        return p_avg_fitness, avg_fitness

    def mutate(self, indi, sig):
        child = Copy_Node(indi)
        sig = sig * np.exp(self.lr*np.random.randn())
        child.mutate(sig, self.bound)
        # child.train(self.data[np.random.choice(len(self.data), self.sub_sample, replace=False)])
        return child, sig

    def crossover(self, p1, p2):
        p1c, p2c = Copy_Node(p1), Copy_Node(p2)
        crossover_pt2 = p2c.locate_crossover()
        if crossover_pt2 is not None:
            if_crossed = p1c.set_crossover(crossover_pt2)
            if if_crossed:
                p1c.train(self.data[np.random.choice(len(self.data), self.sub_sample, replace=False)])
        return p1c

    def selection(self, population, fitness_fuc):
        fitness = []
        for indi in population:
            fitness.append(fitness_fuc(indi))

        fitness = np.array(fitness)
        fit_rank = np.argsort(fitness)[::-1]
        if not self.greedy:
            select_top = np.random.permutation(fit_rank[:self.num_tree])[:int(self.num_tree*0.8)]
            select_rest  =np.random.permutation(fit_rank[self.num_tree:])[:(self.num_tree - int(self.num_tree*0.8))]
            selected = np.concatenate((select_top, select_rest))
        else:
            selected = fit_rank[:self.num_tree]
        # print('new individual count', np.count_nonzero(selected >= self.num_tree))
        # print(np.mean(fitness[:self.num_tree]))
        return selected, np.mean(fitness[:self.num_tree]), np.mean(fitness[selected])

        
class OJrank_wrapper(iForest):
    def __init__(self, data, bound, num_tree, sub_sample = 256):
        self.forest = []
        if len(data) < sub_sample:
            sub_sample = len(data)
        self.l = np.ceil(np.log2(sub_sample))
        self.sub_sample = sub_sample
        self.num_tree = num_tree
        self.bound = bound
        self.data = data
        for _ in range(num_tree):
            sub_data = data[np.random.choice(len(data), sub_sample, replace=False)]
            self.forest.append(Weighted_Node(sub_data, bound, 0, self.l))



class iForest_traced():
    def __init__(self, data, bound, num_tree, sub_sample = 256):
        self.forest = []
        if len(data) < sub_sample:
            sub_sample = len(data)
        self.l = np.ceil(np.log2(sub_sample))
        self.sub_sample = sub_sample
        self.num_tree = num_tree
        self.bound = bound
        for _ in range(num_tree):
            sub_data = data[np.random.choice(len(data), sub_sample, replace=False)]
            self.forest.append(Traced_Node(sub_data, bound, 0, self.l))

    def compute_score(self, pt):
        sum_path_len = 0
        path_len = []
        nodes_id = []
        for tree in self.forest:
            path, node_id = tree.path_len(pt,0)
            sum_path_len += path
            path_len.append(path)
            nodes_id.append(node_id)
        path_len = np.array(path_len)
        return 2**(-sum_path_len/self.num_tree/self.forest[0].avg_external_len(self.sub_sample)), 2**(-path_len/self.forest[0].avg_external_len(self.sub_sample)), nodes_id


    def retrain(self, data):
        if len(data) < sub_sample:
            sub_sample = len(data)
        for i in range(len(self.forest)):
            sub_data = data[np.random.choice(len(data), sub_sample, replace=False)]
            self.forest[i].retrain(sub_data)









