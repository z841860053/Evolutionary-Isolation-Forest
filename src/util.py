import numpy as np

def load_non_temporal(dataset_name):
    from scipy.io import loadmat
    import h5py
    try:
        annots = loadmat('../datasets/'+ '/%s.mat'%dataset_name)
        inp, out = annots['X'], annots['y']
    except:
        annots = h5py.File('../datasets/'+ '/%s.mat'%dataset_name, 'r')
        inp, out = np.array(list(annots['X'])).T, list(annots['y'])[0]
    inp, out = np.array(inp), np.array(out)
    return inp, out

def read_wtg_feedbacks():
    import csv
    import os

    path = '../feedback_weekly'
    labels = []
    need_indices = np.array([0,1,2,-1])

    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.csv'): 

                with open(os.path.join(path, filename)) as csv_file:
                    for row_count, row in enumerate(csv_file):
                        if row_count != 0:
                            labels.append(np.array(row.replace('\n','').split(','))[need_indices])

    return np.array(labels).astype(str)

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() )
    return list_of_objects

def log_func(s1, s2):
    return np.exp(s1 - s2)/(1 + np.exp(s1 - s2))

def from_onehot(v):
    return np.where(v == 1)[0][0]

def to_onehot(idx, len):
    return np.eye(len)[idx]