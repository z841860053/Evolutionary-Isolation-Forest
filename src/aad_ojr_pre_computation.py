import numpy as np 
from iForest import iForest_traced
from util import load_non_temporal
import csv
import os

class Ojr_stream(object):
    """docstring for Ojr_stream"""
    def __init__(self, data):
        self.bound = np.array([np.amin(data, axis=0),np.amax(data, axis=0)], dtype=np.float64).T
        self.iForest = iForest_traced(data, self.bound, 100)
        
    def get_score(self, data):
        score_matrix = []
        node_id_matrix = []
        for pt in data:
            _, score, node_id = self.iForest.compute_score(pt)
            score_matrix.append(score)
            node_id_matrix.append(node_id)

        score_matrix, node_id_matrix = np.array(score_matrix), np.array(node_id_matrix)
        _, nodes_id = np.unique(node_id_matrix.flatten(), return_inverse=True)
        node_number = np.max(nodes_id) + 1
        score_big_matrix = np.zeros((len(data), node_number))
        score_big_matrix[np.repeat(np.arange(len(data)), 100), nodes_id] = score_matrix.flatten()

        return score_big_matrix

    def retrain(self, data):
        self.iForest.retrain(data)



def save_score_list(data, label, ds_name):
    save_dir_orj = './ojrank-master/data/%s/'%ds_name
    if not os.path.exists(save_dir_orj):
        os.mkdir(save_dir_orj)
    save_file_orj = save_dir_orj + '%sTREE.csv'%ds_name
    save_file_aad = './ad_examples-master/ad_examples/datasets/%s_scores.csv'%ds_name

    bound = np.array([np.amin(data, axis=0),np.amax(data, axis=0)], dtype=np.float64).T
    iForest = iForest_traced(data, bound, 100)
    score_matrix = []
    node_id_matrix = []
    score_list_aad = []
    for pt in data:
        _, score, node_id = iForest.compute_score(pt)
        score_matrix.append(score)
        node_id_matrix.append(node_id)
        score_list_aad.append(score)

    score_matrix, node_id_matrix = np.array(score_matrix), np.array(node_id_matrix)
    _, nodes_id = np.unique(node_id_matrix.flatten(), return_inverse=True)
    node_number = np.max(nodes_id) + 1
    score_big_matrix = np.zeros((len(data), node_number))
    score_big_matrix[np.repeat(np.arange(len(data)), 100), nodes_id] = score_matrix.flatten()
    
    if os.path.exists(save_file_orj):
        os.remove(save_file_orj)
    with open(save_file_orj, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        for l, score in zip(label, score_big_matrix):
            writer.writerow(np.append(l,score))

    if os.path.exists(save_file_aad):
        os.remove(save_file_aad)
    with open(save_file_aad, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        first_line = ['label']
        [first_line.append('X%s'%i) for i in range(len(score_list_aad[0]))]
        writer.writerow(first_line)
        for l, score in zip(label, score_list_aad):
            lt = 'nominal' if (l == 0) else 'anomaly'
            writer.writerow(np.append(lt,score))

def save_dataset_aad(data, label, ds_name):
    save_file_aad = './ad_examples-master/ad_examples/datasets/%s.csv'%ds_name
    if os.path.exists(save_file_aad):
        os.remove(save_file_aad)
    with open(save_file_aad, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        first_line = ['label']
        [first_line.append('x%s'%i) for i in range(len(data[0]))]
        writer.writerow(first_line)
        for l, d in zip(label, data):
            lt = 'nominal' if (l == 0) else 'anomaly'
            writer.writerow(np.append(lt,d))

if __name__ == '__main__':
     for (dirpath, dirnames, filenames) in os.walk('../datasets/non-temporal'):
        for filename in filenames:
            if filename.endswith('.mat'):
                ds_name = filename[:-4]
                print(ds_name)
                inp, target = load_non_temporal(ds_name)
                save_dataset_aad(inp, target, ds_name)
                save_score_list(inp, target, ds_name)