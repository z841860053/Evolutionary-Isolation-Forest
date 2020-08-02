import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
import pandas as pd
# from sklearn.ensemble import IsolationForest
# import statsmodels.api as sm

from dataset import Dataset
from iForest import iForest

class Model_1:
    # 1: get trend by moving average
    # 2: predict trend by exponetial weighted slope
    # 3: anomaly score = sqrt(slope**2 + (original-trend)**2)
    def __init__(self, signal, thres):
        self.signal = signal
        self.thres = thres
        self.ma_signal = self.moving_average()

    def moving_average(self):
        max_iter = 4
        avg_range = 15

        ma_signal = []
        for i,signal in enumerate(self.signal):
            it = 0
            #(not self.check_smooth(signal)) & 
            while (it < max_iter):
                signal = np.pad(signal, [avg_range-1,1], 'edge')
                signal = np.mean([signal[k:(k-avg_range)] for k in range(avg_range)], axis=0)
                it += 1           

            ma_signal.append(signal)
        return ma_signal


    def check_smooth(self, signal):
        coe = 0.1
        diff = np.diff(signal)
        precent = np.count_nonzero(diff >= 0)/len(diff) 
        if (precent < coe) | (precent > 1-coe):
            return True
        else:
            return False

    def isolation_forest(self, signal):
        clf = IsolationForest(behaviour='new').fit(signal[:-1].reshape(-1,1))
        return clf.predict([[signal[-1]]])[0]

    def adhoc_detect(self):
        anomaly_idx_ls = []
        for signal, ma, thres in zip(self.signal, self.ma_signal, self.thres):
            diff = np.insert(np.diff(signal),0,0)
            de_trend = signal - ma
            root = np.sqrt(np.array(np.square(diff - np.mean(diff)) + np.square(de_trend)*0.5, dtype=np.float64))
            
            anomaly_idx_ls.append(root)

        return anomaly_idx_ls

    def trend_predict(self, trend):
        trend = trend[-100:]

        w = np.ones(len(trend)-1)*0.3
        for i in range(1,len(w)):
            w[i:] = w[i:]*0.7
        addition = np.sum(np.diff(trend)*w[::-1])
        return addition


class Model_2:
    # isolation forest implementation
    def __init__(self, signal):
        data = self.feautre_extraction(signal)
        self.data = data
        bound1 = np.array([np.amax(data, axis=0)*2,np.amin(data, axis=0)*2], dtype=np.float64).T
        self.iforest = iForest(data, bound1, 100)

    def test(self, signal, start_pt = 0):
        score = []
        lst = []
        data = self.feautre_extraction(signal)
        for pt in data[start_pt:]:
            s, l =self.iforest.compute_score(pt)
            score.append(s)
            lst.append(l)
        return score, lst

    def update(self, signal, fitness_func):
        self.iforest.update(self.feautre_extraction(signal), fitness_func)

    def feautre_extraction(self, signal):
        diff = np.insert(np.diff(signal, axis = 0),0,0, axis=0)
        ma = self.moving_average(signal)
        return np.concatenate((diff, (signal - ma)), axis = 1)

    def moving_average(self, signal):
        max_iter = 1
        avg_range = 5
        it = 0
        #(not self.check_smooth(signal)) & 
        while (it < max_iter):
            sig_pad = []
            for i in range(len(signal[0])):
                sig = signal[:,i]
                sig = np.pad(sig, [avg_range-1,1], 'edge')
                sig_pad.append(sig)
            sig_pad = np.array(sig_pad).T
            signal = np.mean([sig_pad[k:(k-avg_range)] for k in range(avg_range)], axis=0)
            it += 1           

        return signal

    def trend_predict(self, length):
        prediction = self.iforest.predict(self.data, length = length)
        return np.cumsum(prediction, axis = 0)

class Model_3:
    # ARIMA
    def __init__(self, signal):
        # size = int(len(signal) * 0.66)
        # train, test = signal[0:size], signal[size:len(signal)]
        # his = [x for x in signal]
        self.model = auto_arima(signal, start_p=1, start_q=1,
                           max_p=5, max_q=3, m=12,
                           start_P=0, seasonal=False,
                           d=1, D=1, suppress_warnings=True)# trace=True,
                           # error_action='ignore',  
                           # suppress_warnings=True, 
                           # stepwise=True)
        
    def train(self, signal):
        self.model.fit(signal)


    def predict(self, length):
        forcast, intv = self.model.predict(n_periods = length, return_conf_int=True, alpha=0.01)
        return forcast, intv


def weekly_report_iforest(dataset, time_start = '2020-04-09 00:00:00.00'):
    time_start = datetime.strptime(time_start, '%Y-%m-%d %H:%M:%S.%f')
    list_anomaly = []
    header = 'pin_id, quantity_id, generate_time, anomaly_score\n'
    score_lst = []
    show_list = np.array([[21342, 11, '2020-04-15 06:00:00'], [38387, 10, '2020-04-09 12:00:00'], [37937, None, '2020-04-14 00:00:00'], [6109, 10, '2020-04-15 06:00:00'], [40714, 142, '2020-04-09 12:08:29']])

    for i in range(len(show_list)):
        show_list[i,2] = datetime.strptime(show_list[i,2], '%Y-%m-%d %H:%M:%S')
    for group_count, idx_grouped in enumerate(dataset.idx_pin_grouped):
        # print(group_count, len(dataset.idx_pin_grouped))
        signals, threses = dataset.measurement_grouped[idx_grouped], dataset.measurement_group_info[idx_grouped]
        if threses[0][0] in show_list[:,0]:
            print(threses[0][0])
            show_list_idx = np.where(show_list[:,0] == threses[0][0])[0][0]
            time_match = True
            if (len(signals) == 1):
                time_match = False
            else:
                for sig in signals[1:]:
                    if (len(sig[:,0]) != len(signals[0][:,0])):
                        time_match = False
                        break
                    elif (sig[:,0] != signals[0][:,0]).any():
                        time_match = False
                        break

            if time_match:
                not_nonsense_idx = np.where((np.array([(signals[0][i,0] - datetime.now()).total_seconds() for i in range(len(signals[0]))]) < 0))[0]
                signals = np.array([sig[not_nonsense_idx] for sig in signals])
                if (max(signals[0][:,0]) - time_start).total_seconds() > 0:
                    signal_con = np.concatenate([[signal[:,1]] for signal in signals], axis = 0).T

                    start_idx = np.where((np.array([(signals[0][i,0] - time_start).total_seconds() for i in range(len(signals[0]))]) > 0))[0][0]
                    if start_idx > 1:
                        m = Model_2(signal_con[:start_idx])
                        score, _ = m.test(signal_con)

                        for sig, thres in zip(signals, threses):
                            if not np.isinf(thres[2:]).all():
                                m2 = Model_3(sig[:,1])
                                m2.train(sig[:,1])
                                forcast, _ = m2.predict(len(sig)-start_idx)
                                if (forcast < thres[3]).any() | (forcast > thres[4]).any():
                                    print('signal with pin idx:' + str(thres[0]) + ' and quantity idx:' + str(thres[1]) + 'will cross the threshold, thresholds are: ' + str(thres[3]) + 'and' + str(thres[4]))
                        if (max(score[start_idx:]) > 0.5) & (max(score[start_idx:]) == max(score)):
                            anomaly_idx = np.argmax(score[start_idx:]) + start_idx
                            print(threses)
                            # tm = sig[anomaly_idx,0].strftime("%Y-%m-%d, %H:%M:%S")
                            # list_anomaly.append(str(threses[0][0]) + ',' + 'none' + ',' + tm + ',' + str(max(score[(start_idx-1):])) + '\n')
                            # score_lst.append(max(score))
                            fig,a =  plt.subplots(len(signal_con.T)+1,1, figsize=(15, 6), sharex=True)
                            for i, sig in enumerate(signal_con.T):
                                a[i].plot(signals[i][:,0], sig)
                                anomaly_show_idx = np.where(signals[0][:,0] == show_list[show_list_idx, 2])
                                a[i].scatter(signals[i][anomaly_idx,0], sig[anomaly_idx],c='r')
                                diff_in_sec = np.array([np.diff(signals[i][:,0])[j].seconds for j in range(len(np.diff(signals[i][:,0])))])
                                small_int = np.where(diff_in_sec < 10)[0]
                                if np.all(np.diff(signals[i][:,0]) == (signals[i][1,0] - signals[i][0,0])):
                                    period = str(signals[i][1,0] - signals[i][0,0])
                                else:
                                    period =str(np.mean(np.diff(signals[i][:,0])))
                                a[i].set_title('quantity info: ' + dataset.quantities[int(threses[i][1])-1] + '\n period: ' + period, size=8)

                            # idx_max = np.argsort(score)[-2:]
                            # idx_min = np.argsort(score)[0]
                            a[-1].plot(signals[0][start_idx:,0], score[start_idx:])
                            a[-1].set_title('anomaly score')
                            # plt.tight_layout()
                            plt.subplots_adjust(hspace = 0.5,wspace = 1)
                            fig.autofmt_xdate()    
                            plt.show()     
                            # # plt.savefig('plot4.4/%s.png'%count)
                            # # plt.close()

            else:
                for sig, thres in zip(signals, threses):
                    if thres[1] not in show_list[:,1]:
                        continue
                    not_nonsense_idx = np.where((np.array([(sig[i,0] - datetime.now()).total_seconds() for i in range(len(sig))]) < 0))[0]
                    sig = sig[not_nonsense_idx]
                    if (max(sig[:,0]) - time_start).total_seconds() > 0:
                        signal_con = np.array([sig[:,1]]).T
                        start_idx = np.where((np.array([(sig[i,0] - time_start).total_seconds() for i in range(len(sig))]) > 0))[0][0]
                        if start_idx > 1:
                            m = Model_2(signal_con[:start_idx])
                            score, _ = m.test(signal_con)
                            if not np.isinf(thres[2:]).all():
                                m2 = Model_3(sig[:,1])
                                m2.train(sig[:,1])
                                forcast, _ = m2.predict(len(sig)-start_idx)
                                if (forcast < thres[3]).any() | (forcast > thres[4]).any():
                                    print('signal with pin idx:' + str(thres[0]) + ' and quantity idx:' + str(thres[1]) + 'will cross the threshold, thresholds are: ' + str(thres[3]) + 'and' + str(thres[4]))

                            if (max(score[(start_idx):]) > 0.5) & (max(score[(start_idx):]) == max(score)):
                                anomaly_idx = np.argmax(score[(start_idx):]) + start_idx 
                                # tm = sig[anomaly_idx,0].strftime("%Y-%m-%d, %H:%M:%S")
                                # list_anomaly.append(str(thres[0]) + ',' + str(thres[1]) + ',' + tm + ',' + str(max(score[(start_idx-1):])) + '\n')
                                # score_lst.append(max(score))
                                print(thres[2:])
                                fig,a =  plt.subplots(2,1, figsize=(15, 6), sharex=True)
                                a[0].plot(sig[:,0], sig[:,1])
                                anomaly_show_idx = np.where(sig[:,0] == show_list[show_list_idx, 2])[0][0]
                                a[0].scatter(sig[anomaly_idx,0], sig[anomaly_idx,1], c='r')
                                # diff_in_sec = np.array([np.diff(sig[:,0])[j].seconds for j in range(len(np.diff(sig[:,0])))])
                                # small_int = np.where(diff_in_sec < 10)[0]
                                period =str(np.mean(np.diff(sig[:,0])))
                                # a[0].set_title('quantity info: ' + dataset.quantities[int(thres[1])-1] + '\n period: ' + period, size=8)


                                # idx_max = np.argsort(score)[-2:]
                                # idx_min = np.argsort(score)[0]
                                a[-1].plot(sig[start_idx:,0], score[start_idx:])
                                a[-1].set_title('anomaly score')
                                # plt.tight_layout()
                                plt.subplots_adjust(hspace = 0.5,wspace = 1)
                                fig.autofmt_xdate()    
                                plt.show()     

    # list_anomaly = np.array(list_anomaly)
    # sort_idices = np.argsort(np.array(score_lst))
    # list_anomaly = list_anomaly[sort_idices]
    # text_file = open("score.txt", "w")
    # n = text_file.write(header)
    # for l in list_anomaly:
    #     n = text_file.write(l)
    # text_file.close()

if __name__ == '__main__':
    dataset = Dataset(read_csv=False)
    weekly_report_iforest(dataset)
    # count = 0
    # period = timedelta(hours=6)
    # time_allow_error = timedelta(seconds=1)
    # e1,e2,e3 = np.zeros(64), np.zeros(64), np.zeros(64)

    # dataset.plot_per_quantity()
    # for group_count, idx_grouped in enumerate(dataset.idx_pin_grouped):
    #     signals, threses = dataset.measurement_grouped[idx_grouped], dataset.measurement_group_info[idx_grouped]

            # if len(sig) >= 128:
    #             size = 64
    #             train, test = sig[0:-size,1], sig[-size:,1]
    #             m3 = Model_3(train)
    #             m3.train(train)
    #             forecast3,_ = m3.predict(len(test))

    #             m2 = Model_2(np.array([train]).T)
    #             forecast2 = m2.trend_predict(len(test))[:,0] + train[-1]

    #             m1 = Model_1([train], None)
    #             addition = m1.trend_predict(m1.ma_signal[0])
    #             forecast1 = np.arange(len(test))*addition + train[-1]
    #             e1 = e1 + np.abs(forecast1 - test)/max(sig[:,1])
    #             e2 = e2 + np.abs(forecast2 - test)/max(sig[:,1])
    #             e3 = e3 + np.abs(forecast3 - test)/max(sig[:,1])

    # plt.figure(figsize=(12,5), dpi=100)
    # plt.plot(np.arange(len(e1)), e1, label='error naive')
    # plt.plot(np.arange(len(e2)), e2, label='error iforest')
    # plt.plot(np.arange(len(e3)), e3, label='e arima')
    # plt.plot(np.arange(len(test))+len(train) ,test, label='actual')
    # plt.plot(np.arange(len(test))+len(train), forcast2, label='forecast3')
    # # print(forecast3)
    # plt.plot(np.arange(len(test))+len(train), forcast3, label='forecast2')
    # plt.plot(np.arange(len(test))+len(train), forcast1, label='forecast1')
    # plt.fill_between(lower_series.index, lower_series, upper_series, 
    #                  color='k', alpha=.15)
    # plt.title('Forecast vs Actuals')
    # plt.legend(loc='upper left', fontsize=8)
    # plt.show()
        
        # for sig in signals:             
        #     if (len(sig) > 64) & (group_count>1):
        #         if count > 29:
        #             pt_2, pt_3 = [], []
        #             print(group_count)
        #             m3 = Model_3(np.array(sig[0:64,1], dtype=np.float64))
        #             for end_pt in range(64, len(sig)-1):
        #                 # print(end_pt)
        #                 train = sig[0:end_pt,1]
        #                 m2 = Model_2(np.array([train]).T)
        #                 s2, _ = m2.test(np.array([sig[(end_pt-1):(end_pt+1),1]]).T)
        #                 if s2[0] > 0.5:
        #                     pt_2.append(end_pt)
                        
        #                 m3.train(np.array(train, dtype=np.float64))
        #                 _, intv = m3.predict(1)
        #                 if (sig[end_pt,1] > intv[0,1]) | (sig[end_pt,1] < intv[0,0]):
        #                     pt_3.append(end_pt)
        #             pt_2, pt_3 = np.array(pt_2), np.array(pt_3)

        #             fig,a =  plt.subplots(2,1)
        #             a[0].plot(np.arange(64), sig[:64,1])
        #             a[1].plot(np.arange(64), sig[:64,1])
        #             a[0].plot(np.arange(len(sig)-64)+64, sig[64:,1], c = 'b')
        #             a[1].plot(np.arange(len(sig)-64)+64, sig[64:,1], c = 'b')
        #             if len(pt_2) > 0:
        #                 a[0].scatter(np.arange(len(sig))[pt_2], sig[pt_2,1], c='r')
        #             if len(pt_3) > 0:
        #                 a[1].scatter(np.arange(len(sig))[pt_3], sig[pt_3,1], c='r')
                    
        #             # a[1].scatter(m.feautre_extraction(np.array([train]).T), score, c='r')
        #             # a[2].hist(m.feautre_extraction(np.array([train]).T), align='mid')
        #             plt.savefig('arima_vs_iforest/%s.png'%count)
        #             plt.close()
        #         count += 1

                

    # for pin_group_count, idx_pin_grouped in enumerate(dataset.idx_pin_grouped):
    #     signals, threses = dataset.measurement_grouped[idx_pin_grouped], dataset.measurement_group_info[idx_pin_grouped]

    #     time_match = False
    #     if (len(signals) > 1) & (len(signals[0]) > 15):
    #         time_match = True
    #         for sig in signals[1:]:
    #             if (len(sig[:,0]) != len(signals[0][:,0])):
    #                 time_match = False
    #                 break
    #             elif (sig[:,0] != signals[0][:,0]).any():
    #                 time_match = False
    #                 break
    #         if time_match:
    #             signal_con = np.concatenate([[signal[:,1]] for signal in signals], axis = 0).T
               
    #             score = []
    #             m = Model_2(signal_con)
    #             score, lst = m.test(signal_con)

    #             # for i in range(256, len(signal_con)+1):
    #             #     m = Model_2(signal_con[:i])
    #             #     if i == 256:
    #             #         score = m.test(signal_con[:i])
    #             #         break
    #             #     else:
    #             #         score.append(m.test(signal_con[-2+i:i])[-1])
    #             fig,a =  plt.subplots(len(signal_con.T)+1,2, figsize=(15, 6))
    #             for i, sig in enumerate(signal_con.T):
    #                 a[i,0].plot(signals[i][:,0], sig)
    #                 diff_in_sec = np.array([np.diff(signals[i][:,0])[j].seconds for j in range(len(np.diff(signals[i][:,0])))])
    #                 small_int = np.where(diff_in_sec < 10)[0]
    #                 if np.all(np.diff(signals[i][:,0]) == (signals[i][1,0] - signals[i][0,0])):
    #                     period = str(signals[i][1,0] - signals[i][0,0])
    #                 else:
    #                     period = 'unstable, average: ' + str(np.mean(np.diff(signals[i][:,0]))) + '   min: ' + str(np.min(np.diff(signals[i][:,0]))) + '   max: ' + str(np.max(np.diff(signals[i][:,0])))
    #                 a[i,0].set_title('quantity info: ' + dataset.quantities[int(threses[i][1])-1] + '\n period: ' + period, size=8)
                    
    #                 ma = m.moving_average(sig)
    #                 trend_pre = m.trend_predict(ma)
    #                 a[i,1].plot(np.arange(len(ma)), ma)
    #                 a[i,1].plot(np.array([0,40]) + len(ma), np.array([ma[-1] ,trend_pre]), color='r')
    #                 a[i,1].set_title('trend prediction (preliminary)')

    #             idx_max = np.argsort(score)[-2:]
    #             idx_min = np.argsort(score)[0]
    #             a[-1,0].plot(signals[0][:,0], np.insert(score,0,score[0]))
    #             a[-1,0].set_title('anomaly score')
    #             # plt.tight_layout()
    #             plt.subplots_adjust(hspace = 0.5,wspace = 1)
    #             fig.autofmt_xdate()         
    #             plt.savefig('plot4.4/%s.png'%count)
    #             plt.close()

    #     if not time_match:
    #         for sig, thres in zip(signals, threses):
    #             m = Model_2(np.array([sig[:,1]]).T)
    #             score, lst = m.test(np.array([sig[:,1]]).T)
    #             fig,a =  plt.subplots(2,2, figsize=(15, 4))
    #             a[0,0].plot(sig[:,0], sig[:,1])
    #             if np.all(np.diff(sig[:,0]) == (sig[1,0] - sig[0,0])):
    #                 period = str(sig[1,0] - sig[0,0])
    #             else:
    #                 period = 'unstable, average: ' + str(np.mean(np.diff(sig[:,0]))) + '   min: ' + str(np.min(np.diff(sig[:,0]))) + '   max: ' + str(np.max(np.diff(sig[:,0])))
    #             a[0,0].set_title('quantity info: ' + dataset.quantities[int(thres[1])-1] + '\n period: ' + period, size=8)
    #             a[1,0].plot(sig[:,0], np.insert(score,0,score[0]))
    #             a[1,0].set_title('anomaly score')

    #             ma = m.moving_average(sig[:,1])
    #             trend_pre = m.trend_predict(ma)
    #             a[0,1].plot(np.arange(len(ma)), ma)
    #             a[0,1].plot(np.array([0,40]) + len(ma), np.array([ma[-1] ,trend_pre]), color='r')
    #             a[0,1].set_title('trend prediction (preliminary)')
    #             plt.subplots_adjust(hspace = 0.5,wspace = 0.1)
    #             fig.autofmt_xdate()
    #             plt.savefig('plot4.4/%s.png'%count)
    #             plt.close()

    #     count += 1

   
