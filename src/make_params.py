
def make_params():
    params = {}

    params['datasets'] = ['thyroid']
    params['budget'] = 20
    params['num_iter'] = 10

    return params

def make_params_stream():
    params = {}

    params['datasets'] = ['thyroid']
    params['possibility_of_missed_anomaly_be_labeled'] = 0
    params['num_iter'] = 10
    params['anomaly_percent_thres'] = 0.005
    params['max_sample_num'] = 10

    return params