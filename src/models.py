import numpy as np 
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import torch
from torch.nn import functional as F
from torch import nn, optim

from dataset import Dataset
from iForest import iForest

def linear_nomalize(array):
    return (array - min(array))/(max(array) - min(array))

def error_measure(y, p):
    return np.mean(abs(y-p))

class Model_MA:
    # moving average
    def __init__(self, k= 5):
        self.k = k

    def predict_mock(self, signal):
        signal = np.pad(signal, [self.k-1,1], 'edge')
        return np.mean([signal[i:(i-self.k)] for i in range(self.k)], axis=0)

    def raise_anomaly_mock(self, signal):
        prediction = self.predict_mock(signal)
        mae = abs(prediction[:-1]-signal[1:])
        mae = linear_nomalize(mae)
        return np.where(mae > 0.6)[0]

class Model_SES:
    # simple exponential smoothing
    def __init__(self, alpha = 0.3):
        self.alpha = alpha
    
    def predict_mock(self, signal):
        prediction = [signal[0]]
        for i in range(0, len(signal)-1):
            prediction.append(prediction[i]*(1 - self.alpha) + signal[i]*self.alpha)
        return np.array(prediction)

    def raise_anomaly_mock(self, signal):
        prediction = self.predict_mock(signal)
        mae = abs(prediction[:-1]-signal[1:])
        mae = linear_nomalize(mae)
        return np.where(mae > 0.6)[0]

class Model_TES:
    # Talyor's exponential smoothing
    def __init__(self, alpha = 0.3, beta = 0.3):
        self.damp = 0.8
        self.alpha, self.beta = alpha, beta

    def predict_mock(self, signal):
        l, r = [self.alpha*signal[0]], [self.beta]
        prediction = []
        for i in range(0, len(signal)-1):
            p = l[i] * (r[i] ** self.damp)
            prediction.append(p)
            l.append(self.alpha*signal[i+1] + (1-self.alpha)*l[i]*(r[i]**self.damp))
            if l[i] == 0:
                r.append(self.beta + (1-self.beta)*(r[i]**self.damp))
            else:
                r.append(self.beta*l[i+1]/l[i] + (1-self.beta)*(r[i]**self.damp))

        prediction.append(l[-1] * (r[-1] ** self.damp))
        return np.array(prediction)

    def raise_anomaly_mock(self, signal):
        prediction = self.predict_mock(signal)
        mae = abs(prediction[:-1]-signal[1:])
        mae = linear_nomalize(mae)
        return np.where(mae > 0.6)[0]

class Model_Theta:
    # the Theta model
    def __init__(self, alpha = 0.3):
        self.alpha = alpha

    def predict_mock(self, signal):
        diff = np.diff(np.diff(np.pad(signal, [2,0], 'edge')))*2
        predict2 = [diff[0]]
        for i in range(1, len(signal)):
            predict2.append(self.alpha*diff[i] + (1-self.alpha)*predict2[i-1])
        predict2 = np.cumsum(np.cumsum(predict2)) + signal[0]

        predict1 = [signal[0], signal[1]]

        for i in range(2, len(signal)):  
            X = np.array([np.arange(i), np.ones(i)]).T
            y = signal[:i]
            b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            predict1.append(i*b[0] + b[1])

        return (np.array(predict1) + np.array(predict2))/2

    def raise_anomaly_mock(self, signal):
        prediction = self.predict_mock(signal)
        mae = abs(prediction[:-1]-signal[1:])
        mae = linear_nomalize(mae)
        return np.where(mae > 0.6)[0]

class Model_ARIMA:
    # the ARIMA model
    def __init__(self, signal):
        self.model = auto_arima(signal, start_p=1, start_q=1,
                           max_p=5, max_q=3, m=1,
                           start_P=0, seasonal=False,
                           d=1, D=None, suppress_warnings=True)# trace=True,
                           # error_action='ignore',  
                           # suppress_warnings=True, 
                           # stepwise=True)

    def train(self, signal):
        self.model.fit(signal)
    
    def predict(self, signal, length = 1):
        forcast, intv = self.model.predict(n_periods = length, return_conf_int=True, alpha=0.01)
        return forcast, intv

    def raise_anomaly(self, signal):
        pred, intv = self.predict(signal[:-1])
        if (signal[-1] > np.max(intv)) | (signal[-1] < np.min(intv)):
            return True
        else:
            return False

class Model_iForest:

    def __init__(self, signal):
        data = self.feautre_extraction(signal)
        self.data = data
        bound1 = np.array([np.amax(data, axis=0)*2,np.amin(data, axis=0)*2], dtype=np.float64).T
        self.iforest = iForest(data, bound1, 100)

    def test(self, signal, start_pt = 1):
        score = []
        lst = []
        data = self.feautre_extraction(signal)
        for pt in data[(start_pt-1):]:
            s, l =self.iforest.compute_score(pt)
            score.append(s)
            lst.append(l)
        return score, lst

    def feautre_extraction(self, signal):
        diff = np.diff(signal, axis = 0)
        ma = self.moving_average(signal)
        return np.concatenate((diff, (signal - ma)[1:]), axis = 1)

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

class Model_NN(nn.Module):

    def __init__(self):
        super(Model_NN, self).__init__()
        self.network = []
        self.ly1 = nn.Linear(5,10)
        self.ly2 = nn.Linear(10,1)
        params = list(self.ly1.parameters()) + list(self.ly2.parameters())
        self.opt = optim.SGD([p for p in params if p.requires_grad], lr = 0.02, momentum = 0.8)
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.ly1(x))
        return self.ly2(x)

    def train(self, dataset_in, dataset_out):
        for _ in range(6):
            for data_in, data_out in zip(dataset_in, dataset_out):
                data_in, data_out = data_in.unsqueeze(0), data_out.unsqueeze(0)
                self.opt.zero_grad()
                res = self.forward(data_in)
                loss = self.loss_func(res, data_out)
                loss.backward()
                self.opt.step()
                # print(loss.item())

    def test(self, dataset_in):
        with torch.no_grad():
            return self.forward(dataset_in)

    def run_iter(self, signal):
        upper, lower = max(signal), min(signal)
        signal = (signal - lower) / (upper - lower)
        signal = np.array(signal, dtype= np.float)
        data_in = np.array([signal[i:i+5] for i in range(len(signal)-5)])
        data_in = torch.from_numpy(data_in).float()
        data_out = np.array([[signal[i]] for i in range(5, len(signal))])
        data_out = torch.from_numpy(data_out).float()

        sep_pt = len(data_in) * 2//3
        train_in, test_in = data_in[:sep_pt], data_in[sep_pt:]
        train_out, test_out = data_out[:sep_pt], data_out[sep_pt:]

        self.train(train_in, train_out)
        prediction = self.test(data_in)
        prediction = prediction[:,0].numpy()
        prediction = prediction * (upper - lower) + lower
        return prediction


def run_iteratively(signal, model):
    m = model(signal)
    # m.train(signal)
    predict = []
    anomaly = []    
    for i in range(20,len(signal)-1):
        print(i)
        m.train(signal[i-1000:i+1])
        predict.append(m.predict(signal[:i+1])[0])
        anomaly.append(m.raise_anomaly(signal[:i+2]))

    return np.array(predict), np.array(anomaly)


if __name__ == '__main__':
    dataset = Dataset(read_csv=False)
    model_ma, model_ses, model_tes = Model_MA(), Model_SES(), Model_TES()
    model_theta = Model_Theta()
    for group_count, idx_grouped in enumerate(dataset.idx_pin_grouped):
        signals, threses = dataset.measurement_grouped[idx_grouped], dataset.measurement_group_info[idx_grouped]
        print(group_count)
        for signal in signals:
            if len(signal) < 2000:
                continue
            print(len(signal))
            signal = signal[:,1]
            model_nn = Model_NN()
            p_nn = model_nn.run_iter(signal)
            p_ari, ano_ari = run_iteratively(signal, Model_ARIMA)
            p_the = model_theta.predict_mock(signal)
            p_ma, p_ses, p_tes = model_ma.predict_mock(signal), model_ses.predict_mock(signal), model_tes.predict_mock(signal)
            e_ma, e_ses, e_tes = error_measure(p_ma[1:-1], signal[2:]), error_measure(p_ses[1:-1], signal[2:]), error_measure(p_tes[1:-1], signal[2:])
            en_ma, en_ses, en_tes = e_ma/max([e_ma,e_ses,e_tes]), e_ses/max([e_ma,e_ses,e_tes]), e_tes/max([e_ma,e_ses,e_tes])
            plt.plot(np.arange(len(signal)-1), signal[1:], label='original')
            plt.plot(np.arange(len(signal)-1), p_ma[:-1], label='moving average', alpha = 0.5)
            plt.plot(np.arange(len(signal)-1), p_ses[:-1], label='simple exponential smoothing', alpha = 0.5)
            plt.plot(np.arange(len(signal)-1), p_tes[:-1], label='Talyors exponential smoothing', alpha = 0.5)
            plt.plot(np.arange(len(signal)-1), p_the[:-1], label='Theta model', alpha = 0.5)
            plt.plot(np.arange(len(signal)-22)+21, p_ari[:-1], label='arima', alpha = 0.5)
            plt.plot(np.arange(len(p_nn)-1)+5, p_nn[:-1], label='neural network', alpha = 0.5)
            plt.legend(loc='upper left', fontsize=8)
            plt.show()

            



















