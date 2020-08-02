This is the source code for my thesis work: Evolutionary Isolation Forest for Anomaly Detection in A Incremental Scenario.
Experiments for the static scenario and the incremental scenario can be carried out efficiently with the following instrunctions.
The fundaments of Location-based EIF is also in the file EIF_multi_stream.py. However, codes used for WTG experiments are not available thus Location-based EIF is not directly useable.

1. Dependence:
Python 3
Numpy
Matplotlib
Scipy

2. Download ODDS datasets from http://odds.cs.stonybrook.edu/ and put .mat files in folder /dataset/


For the static scenario (classic interactive anomaly detection):

3. modify parameters in function make_params() in make_params.py

4. run $ python evolution.py

5. The result will be saved in /src/result/results_eif_DATASETNAME.csv, where each row represents a run. Each number represents the number of true anomalies found at that certain number of trail.


For the incremental scenario:

3. modify parameters in function make_params_stream() in make_params.py

4. run $ python stream.py

5. The result will be save in /src/result_stream/DATASETNAME/results_pValue_PVALUE_thres_THRES.csv