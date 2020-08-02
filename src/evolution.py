import numpy as np
from iForest import iForest_ES_wrapper, OJrank_wrapper
from util import load_non_temporal
import csv
import os

from make_params import make_params

class Model_iForest_ES():
    def __init__(self, data):
        bound1 = np.array([np.amin(data, axis=0),np.amax(data, axis=0)], dtype=np.float64).T
        self.iForest = iForest_ES_wrapper(data, bound1, 100)
        self.annotated_idx, self.notation = np.array([], dtype = np.int), np.array([], dtype = np.int)
        self.non_annotated_idx = np.arange(len(data))
        self.score_lst_full = None
        self.max_sample_num = 5
        self.low_score_pts_idx, self.high_score_pts_idx = None, None
        self.why = 0

    def raise_anomaly(self):
        score_lst = []
        score_lst_per_tree = []
        for pt in self.iForest.data[self.non_annotated_idx]:
            score, _ = self.iForest.compute_score(pt)
            score_lst.append(score)
        # for pt in self.iForest.data:
        #     score, score_per_tree = self.iForest.compute_score(pt)
        #     score_lst_per_tree.append(score_per_tree)
        # score_lst_per_tree = np.array(score_lst_per_tree)
        # print('bad tree in whole population',self.why)
        # print('bad tree in selected',np.count_nonzero(np.max(score_lst_per_tree, axis = 0) == np.min(score_lst_per_tree, axis = 0)))
        score_rank = np.argsort(score_lst)
        self.low_score_pts_idx = self.non_annotated_idx[score_rank[:20]]
        self.high_score_pts_idx = self.non_annotated_idx[score_rank[-20:]]
        return self.non_annotated_idx[np.argmax(np.array(score_lst))], score_lst

    def fitness_fuc(self, tree):
        idx_same = self.annotated_idx[self.notation == self.notation[-1]][:-1]
        idx_diff = self.annotated_idx[self.notation != self.notation[-1]][-50:]

        fitness = 0
        score_diff = []
        for pt in self.iForest.data[idx_diff]:
            score_diff.append(2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample)))

        lscore, hscore = 1, 0
        if len(idx_diff) > 0:
            for lpt, hpt in zip(self.iForest.data[self.low_score_pts_idx], self.iForest.data[self.high_score_pts_idx]):
                s1 = 2**(-tree.path_len(lpt, 0)/tree.avg_external_len(self.iForest.sub_sample))
                s2 = 2**(-tree.path_len(hpt, 0)/tree.avg_external_len(self.iForest.sub_sample))
                if s1 < lscore:
                    lscore = s1
                if s2 > hscore:
                    hscore = s2

            desire_p = log_func(hscore, lscore)

        pt_self = self.iForest.data[self.annotated_idx[-1]]
        score_self = 2**(-tree.path_len(pt_self, 0)/tree.avg_external_len(self.iForest.sub_sample))
        score_sample = []

        if self.notation[-1] == 1:
            if len(idx_diff) < self.max_sample_num:
                p = 1/self.score_lst_full
                p = p / np.sum(p)
                score_sample = np.random.choice(self.score_lst_full, self.max_sample_num - len(idx_diff), p = p)
            for score in score_diff:
                fitness += desire_p * np.log(log_func(score_self, score)) + (1 - desire_p) * np.log(1 - log_func(score_self, score))
            for score in score_sample:
                p_uv = log_func(score_self, score)
                fitness += (p_uv + 0.1) * np.log(p_uv) + (1 - p_uv - 0.1) * np.log(1 - p_uv)

        else:
            if len(idx_diff) < self.max_sample_num:
                p = (-0.99*self.score_lst_full + 1)**(1/-0.99)
                p = p / np.sum(p)
                score_sample = np.random.choice(self.score_lst_full, self.max_sample_num - len(idx_diff), p = p)
            # else:
            #     continue_neg = self.annotated_idx[(np.where(self.notation != self.notation[-1])[0][-1]+1):]
            #     if len(continue_neg) > 3:
            #         for idx1 in continue_neg:
            #             s2 = score_diff[-1]

            #             s1 = 2**(-tree.path_len(self.iForest.data[idx1], 0)/tree.avg_external_len(self.iForest.sub_sample))
            #             fitness += np.log(log_func(s2, s1))*0.2
            
            for score in score_diff:
                fitness += desire_p * np.log(log_func(score, score_self)) + (1 - desire_p) * np.log(1 - log_func(score, score_self))
            for score in score_sample:
                p_uv = log_func(score, score_self)
                fitness += (p_uv + 0.1) * np.log(p_uv) + (1 - p_uv - 0.1) * np.log(1 - p_uv)


        return fitness

    def fitness_fuc2(self, tree):
        idx_same = self.annotated_idx[self.notation == self.notation[-1]][:-1]
        idx_diff = self.annotated_idx[self.notation != self.notation[-1]][-50:]

        fitness = 0
        score_diff = []
        for pt in self.iForest.data[idx_diff]:
            score_diff.append(2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample)))

        lscore, hscore = 1, 0
        if len(idx_diff) > 0:
            # for lpt, hpt in zip(self.iForest.data[self.low_score_pts_idx], self.iForest.data[self.high_score_pts_idx]):
            #     s1 = 2**(-tree.path_len(lpt, 0)/tree.avg_external_len(self.iForest.sub_sample))
            #     s2 = 2**(-tree.path_len(hpt, 0)/tree.avg_external_len(self.iForest.sub_sample))
            #     if s1 < lscore:
            #         lscore = s1
            #     if s2 > hscore:
            #         hscore = s2

            # desire_p = log_func(hscore, lscore)
            desire_p = 1

        pt_self = self.iForest.data[self.annotated_idx[-1]]
        score_self = 2**(-tree.path_len(pt_self, 0)/tree.avg_external_len(self.iForest.sub_sample))
        score_sample = []

        if self.notation[-1] == 1:
            if len(idx_diff) < self.max_sample_num:
                sample_score_lst = []
                for pt in self.iForest.data[self.low_score_pts_idx]:
                    sample_score_lst.append(2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample)))
                p = 1/np.array(sample_score_lst)
                p = p / np.sum(p)
                score_sample = np.random.choice(sample_score_lst, self.max_sample_num - len(idx_diff), p = p)
            for score in score_diff:
                fitness += desire_p * np.log(log_func(score_self, score))# + (1 - desire_p) * np.log(1 - log_func(score_self, score))
                # print(desire_p, log_func(score_self, score))
            for score in score_sample:
                p_uv = log_func(score_self, score)
                fitness += (p_uv + 0.1) * np.log(p_uv)# + (1 - p_uv - 0.1) * np.log(1 - p_uv)

        else:
            if len(idx_diff) < self.max_sample_num:
                sample_score_lst = []
                for pt in self.iForest.data[self.high_score_pts_idx]:
                    sample_score_lst.append(2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample)))
                p = (-0.99*np.array(sample_score_lst) + 1)**(1/-0.99)
                p = p / np.sum(p)
                score_sample = np.random.choice(sample_score_lst, self.max_sample_num - len(idx_diff), p = p)
            # else:
            #     continue_neg = self.annotated_idx[(np.where(self.notation != self.notation[-1])[0][-1]+1):]
            #     if len(continue_neg) > 3:
            #         for idx1 in continue_neg:
            #             s2 = score_diff[-1]

            #             s1 = 2**(-tree.path_len(self.iForest.data[idx1], 0)/tree.avg_external_len(self.iForest.sub_sample))
            #             fitness += np.log(log_func(s2, s1))*0.2
            
            for score in score_diff:
                # print(desire_p, log_func(score_self, score))
                fitness += desire_p * np.log(log_func(score, score_self))# + (1 - desire_p) * np.log(1 - log_func(score, score_self))
            for score in score_sample:
                p_uv = log_func(score, score_self)
                fitness += (p_uv + 0.1) * np.log(p_uv)# + (1 - p_uv - 0.1) * np.log(1 - p_uv)


        return fitness

    def fitness_fuc3(self, tree):
        idx_same = self.annotated_idx[self.notation == self.notation[-1]][:-1]
        idx_diff = self.annotated_idx[self.notation != self.notation[-1]][-20:]

        fitness = 0
        score_diff = []
        for pt in self.iForest.data[idx_diff]:
            score_diff.append(2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample)))

        pt_self = self.iForest.data[self.annotated_idx[-1]]
        score_self = 2**(-tree.path_len(pt_self, 0)/tree.avg_external_len(self.iForest.sub_sample))
        score_sample = []

        if self.notation[-1] == 1:
            if len(idx_diff) < self.max_sample_num:
                p = 1/self.score_lst_full
                p = p / np.sum(p)
                score_sample = np.random.choice(self.score_lst_full, self.max_sample_num - len(idx_diff), p = p)
            for score in score_diff:
                fitness += np.log(log_func(score_self, score))
            for score in score_sample:
                fitness += np.log(log_func(score_self, score)) * 0.2

        else:
            if len(idx_diff) < self.max_sample_num:
                p = (-0.99*self.score_lst_full + 1)**(1/-0.99)
                p = p / np.sum(p)
                score_sample = np.random.choice(self.score_lst_full, self.max_sample_num - len(idx_diff), p = p)
            else:
                continue_neg = self.annotated_idx[(np.where(self.notation != self.notation[-1])[0][-1]+1):]
                if len(continue_neg) > 3:
                    for idx1 in continue_neg:
                        s2 = score_diff[-1]

                        s1 = 2**(-tree.path_len(self.iForest.data[idx1], 0)/tree.avg_external_len(self.iForest.sub_sample))
                        fitness += np.log(log_func(s2, s1))*0.2
            
            for score in score_diff:
                fitness += np.log(log_func(score, score_self))
            for score in score_sample:
                fitness += np.log(log_func(score, score_self))*0.2


        return fitness


    def update(self, idx, notation, score_lst):
        self.why = 0
        self.score_lst_full = np.array(score_lst)
        self.annotated_idx = np.append(self.annotated_idx, idx)
        self.notation = np.append(self.notation, notation)
        self.non_annotated_idx = np.delete(self.non_annotated_idx, np.where(self.non_annotated_idx == idx)[0])
        self.iForest.update(self.fitness_fuc)



def log_func(s1, s2):
    return np.exp(s1 - s2)/(1 + np.exp(s1 - s2))

def plot_data(data, target):
    import matplotlib.pyplot as plt
    anoamly_idx = np.nonzero(target)
    num_plot = len(data[0])/2
    fig,a =  plt.subplots(num_plot,1)
    for idx in range(num_plot):
        a[idx].scatter(data[:,idx*2], data[:,idx*2+1], s = 5, lw = 0)
        a[idx].scatter(data[anoamly_idx,idx*2], data[anoamly_idx,idx*2+1], s = 10, c='r', lw = 0)

    plt.show()

if __name__ == '__main__':

    params = make_params()

    budget = params['budget']
    num_iter = params['num_iter']

    for (dirpath, dirnames, filenames) in os.walk('../datasets'):
        for filename in filenames:
            if filename.endswith('.mat'):
                dataset_name = filename[:-4]
                # if (dataset_name in ['thyroid', 'lympho', 'optdigits', 'pima','wbc','pendigits','satellite']):
                if (dataset_name not in params['datasets']):
                    continue
                inp, target = load_non_temporal(dataset_name)
                results = []
                save_file = './result/results_eif_%s.csv'%dataset_name
                for _ in range(num_iter):
                    model = Model_iForest_ES(inp)
                    model.iForest.greedy = True
                    true_positive = 0
                    total_trial = 0
                    continue_neg = 0
                    result = []

                    while (total_trial < budget)&(true_positive < np.count_nonzero(target)):

                        idx, score_lst_full = model.raise_anomaly()
                        if target[idx] == 1:
                            true_positive += 1
                            continue_neg = 0
                        # else:
                        #     continue_neg += 1
                        # if continue_neg > 5:
                        #     model.iForest.greedy = False
                        # else:
                        #     model.iForest.greedy = True
                        total_trial += 1
                        print('anomalies found:', true_positive,'  total anomalies:', np.count_nonzero(target), '  total trials:', total_trial)
                        model.update(idx, target[idx], score_lst_full)
                        result.append(true_positive)
                    results.append(result)

                
                    with open(save_file, mode='w') as csv_file:
                        writer = csv.writer(csv_file)
                        for result in results:
                            writer.writerow(result)












