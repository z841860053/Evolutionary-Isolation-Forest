import numpy as np
from iForest import iForest_ES_wrapper, OJrank_wrapper
from util import load_non_temporal
import csv
import os

from make_params import make_params_stream


class Model_iForest_ES():
    def __init__(self, data, params):
        bound1 = np.array([np.amin(data, axis=0),np.amax(data, axis=0)], dtype=np.float64).T
        self.t = 255
        self.iForest = iForest_ES_wrapper(data[:self.t+1], bound1, 100)
        self.annotated_idx, self.notation = np.array([], dtype = np.int), np.array([], dtype = np.int)
        self.score_lst_full = None
        self.max_sample_num = params['max_sample_num']
        self.low_score_pts_idx, self.high_score_pts_idx = None, None
        self.thres = params['anomaly_percent_thres']

    def raise_anomaly(self, data):
        score_lst = []
        for pt in data:
            score, _ = self.iForest.compute_score(pt)
            score_lst.append(score)
        score_rank = np.argsort(score_lst)[::-1]
        for time_step in range(self.t+1, len(score_rank)):
            if np.where(score_rank[score_rank <= time_step] == time_step)[0][0]/len(score_rank[score_rank <= time_step]) < self.thres:
                self.t = time_step
                self.score_lst_full = np.array(score_lst)[:self.t]
                return self.t
        self.score_lst_full = np.array(score_lst)
        return len(score_rank) - 1

    def raise_anomaly2(self,data):
        score_lst = []
        for pt in data:
            score, _ = self.iForest.compute_score(pt)
            score_lst.append(score)
        score_lst = np.array(score_lst)

        ret = np.where(score_lst[self.t+1:] > 0.50)[0]
        if len(ret) == 0:
             return len(score_lst) - 1
        self.t = ret[0] + self.t + 1
        self.score_lst_full = np.array(score_lst)[:self.t]
        return self.t



    def fitness_func(self, tree):
        idx_pos = self.annotated_idx[self.notation == 1]
        idx_neg = self.annotated_idx[self.notation == 0]

        fitness = 0
        score_pos = []
        for pt in self.iForest.data[idx_pos]:
            score_pos.append(2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample)))
        score_pos = np.array(score_pos)

        score_neg = []
        for pt in self.iForest.data[idx_neg]:
            score_neg.append(2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample)))
        score_neg = np.array(score_neg)


        desire_p = 1

        pt_self = self.iForest.data[self.annotated_idx[-1]]
        score_self = 2**(-tree.path_len(pt_self, 0)/tree.avg_external_len(self.iForest.sub_sample))
        score_sample = []

        if self.notation[-1] == 1:
            if len(score_pos) * len(score_neg) < self.max_sample_num:
                p = 1/self.score_lst_full
                p = p / np.sum(p)
                score_sample = np.random.choice(self.score_lst_full, self.max_sample_num - len(score_pos) * len(score_neg), p = p)
                for score in score_sample:
                    p_uv = log_func(score_self, score)
                    fitness += (p_uv + 0.1) * np.log(p_uv) + (1 - p_uv - 0.1) * np.log(1 - p_uv)

        else:
            if len(score_pos) * len(score_neg) < self.max_sample_num:
                p = (-0.99*self.score_lst_full + 1)**(1/-0.99)
                p = p / np.sum(p)
                score_sample = np.random.choice(self.score_lst_full, self.max_sample_num - len(score_pos) * len(score_neg), p = p)

                for score in score_sample:
                    p_uv = log_func(score, score_self)
                    fitness += (p_uv + 0.1) * np.log(p_uv) + (1 - p_uv - 0.1) * np.log(1 - p_uv)

        score_pos_repeat = np.repeat(score_pos, len(score_neg))
        score_neg_repeat = np.tile(score_neg, len(score_pos))

        fitness += np.sum(desire_p * np.log(log_func(score_pos_repeat, score_neg_repeat)))

        return fitness

    def fitness_func2(self, tree):
        idx_pos = self.annotated_idx[self.notation == 1]
        idx_neg = self.annotated_idx[self.notation == 0]

        fitness = 0
        score_pos = []
        for pt in self.iForest.data[idx_pos]:
            score_pos.append(2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample)))
        score_pos = np.array(score_pos)

        score_neg = []
        for pt in self.iForest.data[idx_neg]:
            score_neg.append(2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample)))
        score_neg = np.array(score_neg)


        desire_p = 1

        pt_self = self.iForest.data[self.annotated_idx[-1]]
        score_self = 2**(-tree.path_len(pt_self, 0)/tree.avg_external_len(self.iForest.sub_sample))
        score_sample = []

        if len(score_pos) * len(score_neg) == 0:
            if self.notation[-1] == 1:
                    p = 1/self.score_lst_full
                    p = p / np.sum(p)
                    score_sample = np.random.choice(self.score_lst_full, self.max_sample_num - len(score_pos) * len(score_neg), p = p)
                    for score in score_sample:
                        p_uv = log_func(score_self, score)
                        fitness += (p_uv + 0.1) * np.log(p_uv)

            else:
                    p = (-0.99*self.score_lst_full + 1)**(1/-0.99)
                    p = p / np.sum(p)
                    score_sample = np.random.choice(self.score_lst_full, self.max_sample_num - len(score_pos) * len(score_neg), p = p)

                    for score in score_sample:
                        p_uv = log_func(score, score_self)
                        fitness += (p_uv + 0.1) * np.log(p_uv)
        else:

            score_pos_min = min(score_pos)
            score_neg_max = max(score_pos)

            fitness += np.sum(desire_p * np.log(log_func(score_pos_min, score_neg_max)))

        return fitness

    def fitness_func3(self, tree):
        idx_pos = self.annotated_idx[self.notation == 1]
        idx_neg = self.annotated_idx[self.notation == 0]

        fitness = 0
        score_pos = []
        for pt in self.iForest.data[idx_pos]:
            score_pos = 2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample))
            fitness += min(score_pos, 1) - 1

        for pt in self.iForest.data[idx_neg]:
            score_neg = 2**(-tree.path_len(pt, 0)/tree.avg_external_len(self.iForest.sub_sample))
            fitness += 0.3 - max(score_neg, 0.3)

        return fitness


    def update(self, idx, notation, data):
        self.annotated_idx = np.append(self.annotated_idx, idx)
        self.notation = np.append(self.notation, notation)
        self.iForest.data = data[:self.t+1]

        self.iForest.update(self.fitness_func)



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

    params = make_params_stream()
    p = params['possibility_of_missed_anomaly_be_labeled']



    for (dirpath, dirnames, filenames) in os.walk('../datasets/'):
        for filename in filenames:
            if filename.endswith('.mat'):
                dataset_name = filename[:-4]
                # if (dataset_name in ['thyroid', 'pendigits','optdigits', 'pima', 'wbc', 'lympho']):
                if (dataset_name not in ['thyroid']):
                    continue
                print(dataset_name)
                inp, target = load_non_temporal(dataset_name)
                params['anomaly_percent_thres'] = min([0.05,np.sum(target)/len(target)])
                shuffle  = np.arange(len(target))
                np.random.shuffle(shuffle)
                inp, target = inp[shuffle], target[shuffle]

                if not os.path.isdir('./result_stream/' + dataset_name):
                    os.mkdir('./result_stream/' + dataset_name)

                for i in range(params['num_iter']):

                    save_file = './result_stream/'+ dataset_name +'/results'+'_pValue_' + str(p)+ '_thres_' + str(params['anomaly_percent_thres']) + '.csv'
                    model = Model_iForest_ES(inp, params)
                    model.iForest.greedy = True
                    true_positive = 0
                    total_trial = 0
                    result = ['detected anomalies','total trail', 'total instances', 'total anomalies']

                    pt = 256
                    t = model.raise_anomaly(inp)
                    if target[t] == 1:
                        true_positive += 1
                    total_trial += 1
                    while t < len(inp)-1:
                        remind_idx = np.where(target[pt:t] > 0)[0] + pt
                        remind_idx = remind_idx[np.random.uniform(size=len(remind_idx)) < p]

                        update_idx = np.append(remind_idx, t)
                        model.update(update_idx, target[update_idx], inp)
                        pt = t + 1
                        t = model.raise_anomaly(inp)
                        if target[t] == 1:
                            true_positive += 1
                        total_trial += 1
                        result.append([true_positive, total_trial, t, int(np.sum(target[256:t+1]))])
                        print('recall:' + str(true_positive/np.sum(target[256:t+1])) + '   percision:' + str(true_positive/total_trial),t)
                    final_sort = np.argsort(model.score_lst_full)[::-1]
                    print(np.sum(target[final_sort[:int(np.sum(target))]]), np.sum(target[final_sort[:int(np.sum(target))*2]]))
                
                    with open(save_file, mode='w') as csv_file:
                        writer = csv.writer(csv_file)
                        for line in result:
                            writer.writerow(line)

                    print('anomalies in top n and top 2n: ',[np.sum(target[final_sort[:int(np.sum(target))]]), np.sum(target[final_sort[:int(np.sum(target))*2]])])









