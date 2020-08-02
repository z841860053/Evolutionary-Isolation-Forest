import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from util import load_non_temporal

budget = 200

def convert_string_array(string):
    string  = string[1:-1].replace(' ','')
    return np.array(string.split(','), dtype=np.int)

def read_eif_result(file_name):
    results = []
    with open(file_name) as csvfile:
        csv_file = csv.reader(csvfile)
        for row in csv_file:
            res = np.array(row, dtype = np.int)
            if len(res) < budget:
                res = np.pad(res, (0, budget-len(res)), 'edge')
            results.append(res)

    return results

def read_aad_temp_result(ds_name):
    aad_result = []
    eif_result = []
    count_aad, count_eif = -1, -1
    with open('temp_result.txt', 'r') as txt_file:
        for line in txt_file.readlines():
            line = line[:-1]
            if line == ds_name:
                count_aad = 0
            elif (count_aad < 10) & (count_aad > -1):
                arr = convert_string_array(line)
                if len(arr) < budget:
                    arr = np.pad(arr, (0, budget - len(arr)), 'edge')
                aad_result.append(arr)
                count_aad += 1

            # if line == 'EIF:':
            #     count_eif = 0
            # elif (count_eif < 10) & (count_eif > -1):
            #     array = convert_string_array(line)
            #     if len(array) < 200:
            #         array = np.pad(array, (0, 200-len(array)), 'edge')

            #     eif_result.append(array)
            #     count_eif += 1

    aad_result = np.array(aad_result)
    # eif_result = np.array(eif_result)

    return aad_result#, eif_result

def read_ojrank(dataset_name):
    path = './ojrank-master/result/%s'%dataset_name
    results = []
    for i in range(10):
        res = []
        file_path = os.path.join(path, dataset_name + '_%sTREE.csv'%i)
        with open(file_path) as csvfile:
            csv_file = csv.reader(csvfile)
            for row in csv_file:
                res.append(int(float(row[0])))
        arr = np.array(res)
        if len(arr) < budget:
            arr = np.pad(arr, (0, budget - len(arr)), 'edge')
        results.append(arr)

    return np.array(results)
        
def plot_recall(ds_name):
    total_anomaly = np.count_nonzero(load_non_temporal(ds_name)[1])
    aad_res = read_aad_temp_result(ds_name)
    print(aad_res)
    ojrank_res = read_ojrank(ds_name)
    eif_res = read_eif_result('./result/results_eif_'+ ds_name +'.csv')
    aad_avg, eif_avg, ojrank_avg = np.mean(aad_res, axis = 0), np.mean(eif_res, axis = 0), np.mean(ojrank_res, axis = 0)
    aad_std, eif_std, ojrank_std = np.std(aad_res, axis = 0), np.std(eif_res, axis = 0), np.std(ojrank_res, axis = 0)
    plt.plot(np.arange(len(aad_avg)), aad_avg/total_anomaly, label='AAD')
    plt.plot(np.arange(len(eif_avg)), eif_avg/total_anomaly, label='EIF')
    plt.plot(np.arange(len(ojrank_avg)), ojrank_avg/total_anomaly, label='OJRank')
    plt.legend()
    plt.xlabel('Feedback round')
    plt.ylabel('Recall')
    axes = plt.gca()
    axes.set_xlim([0,50])
    plt.show()

def read_stream_csv(dir, preserve_char = 0):
    results = []
    end_res = []
    hyperparameters = []

    for (dirpath, dirnames, filenames) in os.walk(dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                res = []
                with open(os.path.join(dir, filename)) as csvfile:
                    csv_file = csv.reader(csvfile)
                    for row in csv_file:
                        res.append(np.array(row).astype(float))

                results.append(np.array(res[:-1]))
                end_res.append(np.array(res[-1]))
                hyperparameters.append(filename[-4-preserve_char:-4])

    results = np.array(results)
    end_res = np.array(end_res)

    return results, end_res, hyperparameters

def plot_streaming(ds_name):

    eif_dir = './result_stream/' + ds_name
    results_eif, res_eif_end, _ = read_stream_csv(eif_dir)

    results_eif_end = np.array([res[-1] for res in results_eif])
    eif_percision = np.mean(results_eif_end[:,0])/np.mean(results_eif_end[:,1])
    eif_recall = np.mean(results_eif_end[:,0])/np.mean(results_eif_end[:,3])

    ojr_dir = './ojrank-master/result/' + ds_name + '_stream'
    results_ojr, res_ojr_end, _ = read_stream_csv(ojr_dir)

    results_ojr_end = np.array([res[-1] for res in results_ojr])
    ojr_percision = np.mean(results_ojr_end[:,0])/np.mean(results_ojr_end[:,1])
    ojr_recall = np.mean(results_ojr_end[:,0])/np.mean(results_ojr_end[:,3])
    
    print("%.2f" %eif_percision, "%.2f" %eif_recall)
    print("%.2f" %ojr_percision, "%.2f" %ojr_recall)

    print(np.mean(res_eif_end, axis= 0))
    print(np.mean(res_ojr_end, axis= 0))


def plot_p(ds_name):

    eif_dir = './result_p/' + ds_name
    results_eif, res_eif_full, ps = read_stream_csv(eif_dir, 3)

    # results_eif_end = np.array([res[10] for res in results_eif])

    eif_percision = [np.array(res)[:,0]/np.array(res)[:,1] for res in results_eif]
    eif_recall = [np.array(res)[:,0]/np.array(res)[:,3] for res in results_eif]
    timeline = [np.array(res)[:,2] for res in results_eif]
    # eif_recall = results_eif_end[:,0]/results_eif_end[:,3]
    ps = np.array(ps, dtype = float)

    from scipy.interpolate import interp1d
    intp_precision, intp_recall = [], []
    for res in results_eif:
        res = np.insert(res, 0, np.array([0,0,0,0]), axis = 0)
        f_per = interp1d(np.array(res)[:,2], np.array(res)[:,0]/np.array(res)[:,1])
        intp_pre = f_per(np.arange(res[-1,2]-255) +255)

        f_rec = interp1d(np.array(res)[:,2], np.array(res)[:,0]/np.array(res)[:,3])
        intp_rec = f_per(np.arange(res[-1,2]-255)+255)

        intp_precision.append(intp_pre)
        intp_recall.append(intp_rec)

    intp_precision, intp_recall = np.array(intp_precision), np.array(intp_recall)

    from scipy.stats import pearsonr, kendalltau

    corrs_recall = []
    corrs_precision = []
    for idx in np.linspace(0, len(intp_precision[0]), num = 11)[1:]:
        idx = int(idx)
        pre = intp_precision[:,idx-1]
        corr, _ = kendalltau(ps, pre)
        corrs_precision.append(corr)

        rec = intp_recall[:,idx-1]
        corr, _ = kendalltau(ps, rec)
        corrs_recall.append(corr)

    return corrs_recall

    # print(corrs)
    # from matplotlib import rcParams, cycler
    # cmap = plt.cm.coolwarm
    # rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 11)))
    # from matplotlib.lines import Line2D
    # custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
    #                 Line2D([0], [0], color=cmap(.5), lw=4),
    #                 Line2D([0], [0], color=cmap(1.), lw=4)]

    # fig, ax = plt.subplots()
    # for i in np.argsort(ps):
    #     lines = ax.plot(timeline[i], eif_recall[i])
    # ax.legend(custom_lines, ['Cold', 'Medium', 'Hot'])


    # rgba_colors = np.zeros((len(ps),4))
    # rgba_colors[:,0] = 1.0
    # rgba_colors[:, 3] = ps

    # # print(eif_recall.shape, eif_percision.shape, ps.shape)
    # fig, ax = plt.subplots()
    # im = ax.plot(timeline, eif_percision, c=ps, cmap=plt.cm.jet)
    # fig.colorbar(im, ax=ax)

    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(ds_name)
    plt.show()

def plot_p_cross_ds():
    names = ['pendigits', 'thyroid','mammography']
    corrs = []
    for name in names:
        corr = plot_p(name)
        corrs.append(corr)

    for i in range(3):
        plt.plot(np.linspace(0, 1, num = 11)[1:], corrs[i], label=names[i])
    plt.legend()
    plt.xlabel('Percentage of data that have been processed')
    plt.ylabel("Kendall’s Tau correlation value")
    plt.show()

def plot_percent():
    names = ['thyroid']

    for name in names:
        eif_dir = './result_percent/' + name
        results_eif, res_eif_full, perseved_chars = read_stream_csv(eif_dir, 6)
        perseved_chars = [char.split('_') for char in perseved_chars]
        thres = np.array(perseved_chars, dtype = float)[:,0]

        thres_unique, thres_inv_idx, thres_counts = np.unique(thres, return_inverse=True, return_counts=True)
        precision_per_thres, recall_per_thres = np.zeros(len(thres_unique)), np.zeros(len(thres_unique))
        end_res_per_thres = np.zeros((len(thres_unique),2))

        precision_per_thres, recall_per_thres, end_res_per_thres = \
            np.array(precision_per_thres), np.array(recall_per_thres), np.array(end_res_per_thres)

        for thres_idx, res, res_full in zip(thres_inv_idx, results_eif, res_eif_full):
            percision = res[-1,0]/res[-1,1]
            recall = res[-1,0]/res[-1,3]
            precision_per_thres[thres_idx] += percision
            recall_per_thres[thres_idx] += recall
            end_res_per_thres[thres_idx][0] += res_full[0]
            end_res_per_thres[thres_idx][1] += res_full[1]

        

        precision_per_thres /= thres_counts
        recall_per_thres /= thres_counts
        end_res_per_thres[:,0] /= thres_counts
        end_res_per_thres[:,1] /= thres_counts

        print(end_res_per_thres/93)

        for per, rec, thres in zip(precision_per_thres, recall_per_thres, thres_unique):
            plt.scatter(per, rec, label = thres)
        plt.legend(title="threshold")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('Precision')
        plt.ylabel("Recall")
        plt.show()

def plot_wtg_different_method():

    resss = []

    for i in range(5):
        ress = []
        for j in range(2):
            pn = 'pos' if j else 'neg'
            filename = './result_wtg/'+pn+'_rank' + str(i)
            res = []

            with open(filename) as csvfile:
                csv_file = csv.reader(csvfile)
                for row in csv_file:
                    res.append(np.array(row).astype(float).astype(int))

            ress.append(np.concatenate(res))
        resss.append(ress)

    resss = np.array(resss)
    titles = ['Naive Isolation Forest', 'Location-based Isolation Forest', 
        'Location-based grouped EIF', 'Location-based EIF re-rank', 'Location-based EIF']

    for i, (neg, pos) in enumerate(resss):
        print(titles[i])
        # pos = pos[pos <10]
        # neg = neg[neg <10]
        top1_acc = np.sum(pos < 1)/(np.sum(pos < 1) + np.sum(neg < 1))
        top5_acc = np.sum(pos < 5)/(np.sum(pos < 5) + np.sum(neg < 5))
        top20_acc = np.sum(pos < 20)/(np.sum(pos < 20) + np.sum(neg < 20))
        print(1-top1_acc, 1-top5_acc, 1-top20_acc)
        # bins=np.histogram(np.hstack((pos,neg)), bins=50)[1]
        # # bins = [1,2,3,4,5,7.5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,200]
        # # print(bins)
        # plt.hist(pos, density=True, bins=bins, alpha = 0.7, label = 'Anomalies')
        # plt.hist(neg, density=True, bins=bins, alpha = 0.7, label = 'Normal Instances')
        # plt.ylabel('Density')
        # plt.xlabel('Rank')
        # plt.legend()
        # plt.title(titles[i])
        
        # plt.show()






if __name__ == '__main__':
    # test()
    # names = ['annthyroid', 'arrhythmia', 'cardio', 
    #     'ionosphere', 'letter', 'mammography', 'mnist', 
    #     'musk', 'pendigits', 'pima', 'satellite', 
    #     'satimage-2', 'speech', 'thyroid', 'vowels']
    plot_wtg_different_method()













