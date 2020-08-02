import numpy as np 
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from datetime import datetime, timedelta
from scipy import stats

class Dataset:
    def __init__(self, read_csv = False, interpolate = True):

        self.dominant_peroid = []

        if read_csv:

            self.load_measurements()
            self.load_thresholds()
            self.load_quatities()
            combine_id_thres = np.core.defchararray.add(self.thresholds[0], self.thresholds[1])
            self.measurement_grouped = []
            self.measurement_group_info = []
            for idx_l1 in self.idx_time_grouped:
                self.measurement_grouped.append(self.measurement[np.array([1,-1])][:,idx_l1])

                combine_id_meas = np.core.defchararray.add(self.measurement[0,idx_l1[0]], self.measurement[2,idx_l1[0]])
                idx_combine_id = np.where(combine_id_thres == combine_id_meas)[0]
                if len(idx_combine_id) != 0:
                    clow, low, high, chigh = self.thresholds[2:,idx_combine_id[0]]
                else:
                    clow, low, high, chigh = -np.inf, -np.inf, np.inf, np.inf
                # print(self.measurement[0,idx_l1[0]])
                info = np.array([self.measurement[0,idx_l1[0]], self.measurement[2,idx_l1[0]], clow, low, high, chigh])
                self.measurement_group_info.append(info)

            for i in range(len(self.measurement_grouped)):
                temp = self.measurement_grouped[i][0]
                temp2 = self.measurement_grouped[i][1].astype(float)
                temp3 = []
                for t in temp:
                    try :
                        temp3.append(datetime.strptime(t, '%Y-%m-%d %H:%M:%S'))
                    except:
                        temp3.append(datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f'))
                self.measurement_grouped[i] = np.array([temp3, temp2]).T


            for i in range(len(self.measurement_group_info)):
                self.measurement_group_info[i] = self.measurement_group_info[i].T.astype(float)

            self.measurement_grouped = self.interpolate(self.measurement_grouped)
            self.measurement_group_info = np.array(self.measurement_group_info)
            self.quantities = np.array(self.quantities)

            if True:
                np.save('temp/measurement_grouped.npy', self.measurement_grouped)
                np.save('temp/measurement_group_info.npy', self.measurement_group_info)
                np.save('temp/quantities_info.npy', self.quantities)
        else:
            self.measurement_grouped = np.load('temp/measurement_grouped.npy', allow_pickle=True)
            self.measurement_group_info = np.load('temp/measurement_group_info.npy', allow_pickle=True)
            self.quantities = np.load('temp/quantities_info.npy', allow_pickle=True)

            # self.measurement_grouped = self.interpolate(self.measurement_grouped)

            # diff_lst = self.dominant_peroid
            # period_in_sec = np.array([x.total_seconds() for x in diff_lst])/3600
            # weights = np.ones_like(period_in_sec) / len(period_in_sec)

            # plt.hist(period_in_sec, range=[0,24], color = 'blue', edgecolor = 'black', bins = 100, weights=weights)
            # # axes = plt.gca()
            # # axes.set_ylim([0,0.15])
            # plt.title('distribution of period of differet signals')
            # plt.xlabel('period (hours)')
            # plt.ylabel('percentage')
            # plt.show()

        pin_id_uni, pin_id_inv = np.unique(self.measurement_group_info[:,0], return_inverse = True)
        idx_pin_grouped = []
        for i, id_uni in enumerate(pin_id_uni):
            idx = np.where(pin_id_inv == i)[0]
            idx_pin_grouped.append(idx)
        self.idx_pin_grouped = np.array(idx_pin_grouped)

        quantity_id_uni, quantity_id_inv = np.unique(self.measurement_group_info[:,1], return_inverse = True)
        idx_quantity_grouped = []
        for i, id_uni in enumerate(quantity_id_uni):
            idx = np.where(quantity_id_inv == i)[0]
            idx_quantity_grouped.append(idx)
        self.idx_quantity_grouped = np.array(idx_quantity_grouped)
        
    def interpolate(self, measurement_old):
        measurements = []
        for sig in measurement_old:
            if len(sig) < 3:
                continue
            time_diff = np.diff(sig[:,0])
            idx_keep = np.append(0, np.where(np.array([x.total_seconds() for x in time_diff]) > 1)[0] + 1)
            sig = sig[idx_keep]
            if len(sig) < 3:
                continue
            time_diff = np.diff(sig[:,0])

            dominant_peroid = np.median(time_diff)#.total_seconds()
            # min_time_allowance = timedelta(seconds=dominant_peroid*0.01)
            idx_keep = np.append(0, np.where(time_diff > dominant_peroid*0.05)[0] + 1)
            sig = sig[idx_keep]
            if len(sig) < 3:
                continue
            measurements.append(sig)
            time_diff = np.diff(sig[:,0])
        
            self.dominant_peroid.append(dominant_peroid)
        
        return np.array(measurements)

    def load_measurements(self, path = None, name_column=None, name_must_have=None):
        path = '../WTG_Dataset/200309 measurements.csv'
        name_column = ['pin_id', 'generated_at', 'client_quantity_id', 'order_of_magnitude', 'significand']
        name_must_have = 'pin_id'
        check_idx = [0,2]
        time_idx = 2

        name_column = np.array(name_column)
        idx_must_not_empty = np.where(name_column == name_must_have)[0]
        measurement1 = self.load_csv(path, name_column)
        
        path2 = '../WTG_Dataset/measurements2.csv'
        name_column2 = ['pinId', 'generatedAt', 'clientQuantityId', 'orderOfMagnitude', 'significand']
        name_column2 = np.array(name_column2)
        measurement2 = self.load_csv(path2, name_column2)

        measurements = [measurement1, measurement2]
        for i in range(3,11):
            path = '../WTG_Dataset/measurements%s.csv'%i
            measurements.append(self.load_csv(path, name_column2))


        # for i in range(len(measurement2[1])):
        #     measurement2[1,i] = measurement2[1,i] + '.000'
        # print(measurement2[1])
        # measurement = measurement1
        measurement = np.concatenate(measurements, axis=1)
        print('load csv finished', measurement.shape)
        measurement = measurement[:,np.where((measurement[idx_must_not_empty] != 'NULL') & (measurement[idx_must_not_empty] != 'null'))[1]]
        measurement[-2] = np.array(measurement[-1], dtype=np.float) * (10 ** np.array(measurement[-2], dtype=np.float))
        measurement = np.delete(measurement, -1, 0)

        measurement[check_idx[0]] = np.char.zfill(measurement[check_idx[0]], 6)
        measurement[check_idx[1]] = np.char.zfill(measurement[check_idx[1]], 3)
        combine_id = np.core.defchararray.add(measurement[check_idx[0]], measurement[check_idx[1]])
        id_unique, idx_inverse = np.unique(combine_id, return_inverse=True)
        idx_time_grouped = []
        for i, id_uni in enumerate(id_unique):
            idx = np.where(idx_inverse == i)[0]
            if (len(idx) > 1) & (len(np.unique(measurement[-1,idx])) > 1):
                idx_time_grouped.append(idx)

        # remove_idx_ls = np.array([])
        # for idx_idx, idx in enumerate(idx_time_grouped):
            # xxx, tm_idx_inverse, tm_idx_count = np.unique(measurement[time_idx, idx], return_inverse=True, return_counts=True)
            # repeat = np.where(tm_idx_count != 1)[0]

            # if len(repeat) != 0:
            #     for rep in repeat:
            #         rep_idx = np.where(tm_idx_inverse == rep)[0]
            #         values = np.array(measurement[-1, idx[rep_idx]], dtype=np.float64)
            #         remove_idx = np.where(values == 0)[0] 
            #         if len(repeat) - len(remove_idx) == 1:
            #             # remove_idx_ls = np.append(remove_idx_ls, idx[remove_idx])
            #             idx_time_grouped[idx_idx] = np.delete(idx_time_grouped[idx_idx], remove_idx, 0)
            #             # print('noise input found at index:' + str(remove_idx))
            #         else:
            #             # remove_idx_ls = np.append(remove_idx_ls, idx[rep_idx[1:]])
            #             idx_time_grouped[idx_idx] = np.delete(idx_time_grouped[idx_idx], idx[rep_idx[1:]], 0)
            #             # print('repeat input found at index:' + str(idx[rep_idx]))
            #             # print(measurement[1:4, idx[rep_idx[0]]].T)
         
        # self.measurement = np.delete(measurement, remove_idx_ls, 1)
        self.measurement = np.array(measurement)
        self.idx_time_grouped = np.array(idx_time_grouped)
        print('data clean finished')


    def load_thresholds(self):
        # threshold: 1: pin id 2: quantity id 3: c low 4: low 5: high 6: c high
        path = '../WTG_Dataset/200309 thresholds.csv'
        name_column = ["quantity_id","critically_low_order_of_magnitude","critically_low_significand",
        "low_order_of_magnitude","low_significand","high_order_of_magnitude","high_significand",
        "critically_high_order_of_magnitude","critically_high_significand","pin_id"]
        check_idx = [0,1]

        name_column = np.array(name_column)
        thresholds = self.load_csv(path, name_column)
        for i in range(2,9,2):
            idx = np.where(thresholds[i] == 'NULL')[0]
            thresholds[i, idx] = np.inf * np.sign(i-5)
        # thresholds = np.array(thresholds, dtype=np.float)
        self.thresholds = np.zeros([6, thresholds.shape[1]])
        self.thresholds = self.thresholds.astype(str)
        self.thresholds[1], self.thresholds[0] = thresholds[0], thresholds[-1]
        for i in range(2,9,2):
            self.thresholds[i//2+1] = (thresholds[i].astype(float) * 10 ** thresholds[i-1].astype(float)).astype(str)
        self.thresholds[check_idx[0]] = np.char.zfill(np.array(self.thresholds[check_idx[0]], dtype=np.str), 6)
        self.thresholds[check_idx[1]] = np.char.zfill(np.array(self.thresholds[check_idx[1]], dtype=np.str), 3)

    def load_quatities(self):
        path = '../WTG_Dataset/200318 quantities production.csv'
        name_column = ["id","name","unit","quantity_key"]

        name_column = np.array(name_column)
        quantities = self.load_csv(path, name_column)
        x1 = np.core.defchararray.add(quantities[1], np.array([',']*len(quantities[1])))
        x2 = np.core.defchararray.add(quantities[2], np.array([',']*len(quantities[2])))

        self.quantities = np.empty(max(np.array(quantities[0], dtype=np.int))+1, dtype = quantities.dtype)
        self.quantities[np.array(quantities[0], dtype=np.int)] = np.core.defchararray.add(np.core.defchararray.add(x1,x2), quantities[2]).T

    def group_by(self, group_idx, lower_layer_groups):
        ls = self.measurement[group_idx, [idx[0] for idx in lower_layer_groups]]
        search_id, inv = np.unique(ls, return_inverse=True)
        higher_layer_groups = []
        for i in range(len(search_id)):
            higher_layer_groups.append(np.where(inv == i)[0])

        return search_id, higher_layer_groups


    def plot(self):

        quantity_names = []
        quantity_counts = []
        quantity_signals = []

        for group_count, idx_grouped in enumerate(self.idx_quantity_grouped):

            idx = np.random.choice(idx_grouped,1)[0]
            sig = self.measurement_grouped[idx]
            info = self.measurement_group_info[idx]

            quantity = self.quantities[int(info[1])]
            if quantity in quantity_names:
                quantity_counts[quantity_names.index(quantity)] += len(idx_grouped)

            else:
                quantity_names.append(quantity[:quantity.find(',',quantity.find(',')+1)])
                quantity_counts.append(len(idx_grouped))
                quantity_signals.append(sig)

        quantity_names = np.array(quantity_names)
        quantity_counts = np.array(quantity_counts)
        quantity_signals = np.array(quantity_signals)

        sort_count = np.argsort(quantity_counts)[::-1]

        fig,a =  plt.subplots(4,3)
        for i, (name, sig) in enumerate(zip(quantity_names[sort_count], quantity_signals[sort_count])):
            a[i//3, i%3].plot(sig[:,0], sig[:,1])
            a[i//3, i%3].set_title(name)
            if i == 11:
                break

        plt.subplots_adjust(hspace=0.4)
        plt.setp(a, xticks=[])
        plt.show()


    def distribution_test(self):
        quantity_names = []
        quantity_idices = []
        quantity_counts = []
        for group_count, idx_grouped in enumerate(self.idx_quantity_grouped):

            quantity_name = self.quantities[int(self.measurement_group_info[idx_grouped[0]][1])]
            if quantity_name in quantity_names:
                quantity_idices[quantity_names.index(quantity_name)] =\
                    np.append(quantity_idices[quantity_names.index(quantity_name)], idx_grouped)
                quantity_counts[quantity_names.index(quantity_name)] += len(idx_grouped)
            else:
                quantity_names.append(quantity_name)
                quantity_idices.append(idx_grouped)
                quantity_counts.append(len(idx_grouped))

        quantity_names = np.array(quantity_names)
        quantity_counts = np.array(quantity_counts)
        quantity_idices = np.array(quantity_idices)

        sort_count = np.argsort(quantity_counts)[::-1]

        for name, idices in zip(quantity_names[sort_count], quantity_idices[sort_count]):
            print(name, len(idices))
            np.random.shuffle(idices)
            p_values = []
            count = 0
            for i, idx1 in enumerate(idices):
                count2 = 0
                for idx2 in idices[i+1:]:
                    sig1 = self.measurement_grouped[idx1]
                    sig2 = self.measurement_grouped[idx2]
                    _, p = stats.ks_2samp(sig1[:,1], sig2[:,1])
                    p_values.append(p)
                    count += 1
                    count2 += 1
                    if count >= 10:
                        break
                if count2 >= 100:
                    break

            print(p_values)
            print('max p: ', max(p_values), '   min p: ', min(p_values))
            print('average p: ', np.mean(p_values))
            exit()

        # for group_count, idx_grouped in enumerate(self.idx_quantity_grouped):
        #     quantity_name = self.quantities[int(self.measurement_group_info[idx_grouped[0]][1])]
        #     signals = self.measurement_grouped[idx_grouped]

        #     if len(signals) < 2:
        #         continue

        #     print(quantity_name, len(signals))

        #     p_values = []
        #     count = 0

        #     fig,a =  plt.subplots(10,3)

        #     for i, sig1 in enumerate(signals):
        #         a[i//3, i%3].plot(sig1[:,0], sig1[:,1])
        #         if i == 29:
        #             break
        #     plt.setp(a, xticks=[])
        #     plt.show()
                # if len(sig1) < 100:
                #     continue
                # count2 = 0
                # for sig2 in signals[i+1:]:
                #     if (len(sig2) < 100) | np.array_equal(sig1[:,1], sig2[:,1]):
                #         continue
                #     _, p = stats.ks_2samp(sig1[:,1], sig2[:,1])
                #     p_values.append(p)

                #     count += 1
                #     count2 += 1
                #     if count >= 10:
                #         break
                # if count2 >= 100:
                #     break
            
            # print('max p: ', max(p_values), '   min p: ', min(p_values))
            # print('average p: ', np.mean(p_values))


    def load_csv(self, path, name_column):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            register = None
            data = []
            count = 0
            for row_count, row in enumerate(csv_reader):
                # if row_count > 400000:
                #     break
                if row_count == 0:
                    register = [0] * len(name_column)

                    for element_count, element in enumerate(row):
                        position = np.nonzero(name_column == element)[0]
                        if len(position) !=0:
                            register[position[0]] = element_count

                else:
                    row = np.array(row)
                    data_row = row[register]
                    data.append(data_row)
            data = np.array(data)
            return data.T

    def plot_time_diff_distribution(self, period = timedelta(hours=6)):
        time_allow_error = timedelta(seconds=60)
        diff_lst = np.array([])
        for sig in self.measurement_grouped:
            diff = np.diff(sig[:,0])
            if abs(np.median(diff) - period) < time_allow_error:
                diff_lst = np.append(diff_lst, diff)

        weights = np.ones_like(diff_lst) / len(diff_lst)
        plt.hist([x.seconds for x in diff_lst], range=[0,3600*24], color = 'blue', edgecolor = 'black', bins = 100, weights=weights)
        # axes = plt.gca()
        # axes.set_ylim([0,0.15])
        plt.title('distribution of difference for signal that have period of : %s'%period)
        plt.xlabel('period (seconds)')
        plt.ylabel('percentage')
        plt.show()

    def plot_period_distribution(self):
        period = []
        min_date = datetime.strptime('2020-01-01 00:00:00.00', '%Y-%m-%d %H:%M:%S.%f')
        for sig in self.measurement_grouped:
            period.append(np.mean(np.diff(sig[:,0])).total_seconds())
        period = np.array(period)
        print(min(period), max(period))
        weights = np.ones_like(period) / len(period)
        plt.hist(period, color = 'blue', range = [0,3600*24], edgecolor = 'black', bins = 100, weights=weights)
        # axes = plt.gca()
        # axes.set_ylim([0,0.15])
        plt.title('distribution of period across the dataset')
        plt.xlabel('period (seconds)')
        plt.ylabel('percentage')
        plt.show()

    def features(self):
        min_date = datetime.strptime('2020-01-01 00:00:00.00', '%Y-%m-%d %H:%M:%S.%f')
        max_date = datetime.strptime('1990-01-01 00:00:00.00', '%Y-%m-%d %H:%M:%S.%f')
        for sig in self.measurement_grouped:
            if min(sig[:,0]) < min_date:
                min_date = min(sig[:,0])
            if max(sig[:,0]) > max_date:
                max_date = max(sig[:,0])

        print('max min date', min_date, max_date)
        print('number signals:', len(self.measurement_grouped))
        print('number pins:', len(self.idx_pin_grouped))

        quantity_names = []
        for group_count, idx_grouped in enumerate(self.idx_quantity_grouped):
            
            signals, threses = self.measurement_grouped[idx_grouped], self.measurement_group_info[idx_grouped]
            quantity_names.append(self.quantities[int(threses[0][1])])
        print('quantities:', np.unique(quantity_names))
        print('number unique quantities:', len(np.unique(quantity_names)))

    def plot_per_quantity(self):
        for group_count, idx_grouped in enumerate(self.idx_quantity_grouped):
            if len(idx_grouped) > 30:
                signals, threses = self.measurement_grouped[idx_grouped], self.measurement_group_info[idx_grouped]
                size = np.array([len(sig) for sig in signals])
                if len(signals[size > 1000])> 3:
                    fig,a =  plt.subplots(3,1,figsize=(10,6))
                    for i, sig in enumerate(signals[size > 1000][np.random.choice(len(signals[size > 1000]),3)]):
                        a[i].plot(np.arange(len(sig)), sig[:,1])
                    a[0].set_title('quantity:' + self.quantities[int(threses[0][1])])
                    plt.show()
                # mean_lst = [np.mean(np.diff(sig[:,1])) for sig in signals]
                # std_lst = [np.std(np.diff(sig[:,1])) for sig in signals]
                # print(stats.shapiro(mean_lst))
                # weights = np.ones_like(mean_lst) / len(mean_lst)
                # fig,a =  plt.subplots(2,1)
                # a[0].hist(mean_lst, color = 'blue', edgecolor = 'black', bins = 100, weights=weights)
                # a[0].set_title('distribution of means of all signals from the same quantity')
                # a[1].hist(std_lst, color = 'blue', edgecolor = 'black', bins = 100, weights=weights)
                # a[1].set_title('distribution of standard deviation of all signals from the same quantity')
                # plt.show()


class Dataset_new(object):

    def __init__ (self, read_csv = True):
        if read_csv:
            self.dominant_peroid = []
            self.load_measurements()

            self.measurement_grouped = []
            self.measurement_group_info = []
            for idx_l1 in self.idx_time_grouped:
                self.measurement_grouped.append(self.measurement[np.array([1,-1])][:,idx_l1])

                info = np.array([self.measurement[0,idx_l1[0]], self.measurement[2,idx_l1[0]]])
                self.measurement_group_info.append(info)

            for i in range(len(self.measurement_grouped)):
                temp = self.measurement_grouped[i][0]
                temp2 = self.measurement_grouped[i][1].astype(float)
                temp3 = []
                for t in temp:
                    try :
                        temp3.append(datetime.strptime(t, '%Y-%m-%d %H:%M:%S'))
                    except:
                        temp3.append(datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f'))
                self.measurement_grouped[i] = np.array([temp3, temp2]).T


            for i in range(len(self.measurement_group_info)):
                self.measurement_group_info[i] = self.measurement_group_info[i].T.astype(float)

            self.measurement_grouped = self.interpolate(self.measurement_grouped)
            self.measurement_group_info = np.array(self.measurement_group_info)

        pin_id_uni, pin_id_inv = np.unique(self.measurement_group_info[:,0], return_inverse = True)
        idx_pin_grouped = []
        for i, id_uni in enumerate(pin_id_uni):
            idx = np.where(pin_id_inv == i)[0]
            idx_pin_grouped.append(np.array(idx))
        self.idx_pin_grouped = np.array(idx_pin_grouped)

    def interpolate(self, measurement_old):
        measurements = []
        for sig in measurement_old:
            if len(sig) < 3:
                continue
            time_diff = np.diff(sig[:,0])
            idx_keep = np.append(0, np.where(np.array([x.total_seconds() for x in time_diff]) > 1)[0] + 1)
            sig = sig[idx_keep]
            if len(sig) < 3:
                continue
            time_diff = np.diff(sig[:,0])

            dominant_peroid = np.median(time_diff)#.total_seconds()
            # min_time_allowance = timedelta(seconds=dominant_peroid*0.01)
            idx_keep = np.append(0, np.where(time_diff > dominant_peroid*0.05)[0] + 1)
            sig = sig[idx_keep]
            if len(sig) < 3:
                continue
            measurements.append(sig)
            time_diff = np.diff(sig[:,0])
        
            self.dominant_peroid.append(dominant_peroid)
        
        return np.array(measurements)

    def load_measurements(self, path = None, name_column=None, name_must_have=None):
        path = '../WTG_Dataset/special2.csv'
        name_column = ['pinId', 'generatedAt', 'clientQuantityId', 'orderOfMagnitude', 'significand']
        name_must_have = 'pinId'
        check_idx = [0,2]
        time_idx = 2

        name_column = np.array(name_column)
        idx_must_not_empty = np.where(name_column == name_must_have)[0]
        measurement = self.load_csv(path, name_column)
        

        # measurement = np.concatenate(measurements, axis=1)
        print('load csv finished', measurement.shape)
        measurement = measurement[:,np.where((measurement[idx_must_not_empty] != 'NULL') & (measurement[idx_must_not_empty] != 'null'))[1]]
        measurement[-2] = np.array(measurement[-1], dtype=np.float) * (10 ** np.array(measurement[-2], dtype=np.float))
        measurement = np.delete(measurement, -1, 0)

        measurement[check_idx[0]] = np.char.zfill(measurement[check_idx[0]], 6)
        measurement[check_idx[1]] = np.char.zfill(measurement[check_idx[1]], 3)
        combine_id = np.core.defchararray.add(measurement[check_idx[0]], measurement[check_idx[1]])
        id_unique, idx_inverse = np.unique(combine_id, return_inverse=True)
        idx_time_grouped = []
        for i, id_uni in enumerate(id_unique):
            idx = np.where(idx_inverse == i)[0]
            if (len(idx) > 1) & (len(np.unique(measurement[-1,idx])) > 1):
                idx_time_grouped.append(idx)

        self.measurement = np.array(measurement)
        self.idx_time_grouped = np.array(idx_time_grouped)
        print('data clean finished')

    def load_csv(self, path, name_column):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            register = None
            data = []
            count = 0
            for row_count, row in enumerate(csv_reader):
                # if row_count > 400000:
                #     break
                if row_count == 0:
                    register = [0] * len(name_column)

                    for element_count, element in enumerate(row):
                        position = np.nonzero(name_column == element)[0]
                        if len(position) !=0:
                            register[position[0]] = element_count

                else:
                    row = np.array(row)
                    data_row = row[register]
                    data.append(data_row)
            data = np.array(data)
            return data.T


if __name__ == '__main__':
    dataset = Dataset()
    # dataset.load_measurements()
    dataset.distribution_test()
    

            




