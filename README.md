# Indoor Location System Using FingerPrint and Machine Learning

We compared different algorithms to estimate user's outdoor position using the RSSIs (Received Signal Strength Indicator) collected from different RBS (Radio Base Station) by user's mobile device.

Usually, the FingerPrint [1] algorithm is used to perform this task. Thus, we evaluated the Fingerprint's performance and compared it with many machine learning models such as kNN, SVM, and Random Forest. These algorithms were used to do a regression of user's position based on the RSSI's values.

The analyzed dataset (datasets/medicoes.csv) contains about 2500 lines with 6 different measures from 6 different RBSs. Also, it has 2 columns which represent the user's position (our target). The erbs.csv dataset provides more information about the RBSs such as its positions and the power of the radiated signal. This project was developed in Python using Pandas [2] and scikit-learn [3] libraries.

# Results

### FingerPrint ###

We implemented the algorithm and many theoretical propagation models which are used to create the cover map used by the algorithm.
More information about the algorithm can be found at [1]. 

	$ python run_fingerprint.py

The following results are the best average error accomplished by the naive fingerprint implementation plus the theoretical propagation models.

![finger-print](https://github.com/marcelsan/commoveis/blob/master/outputs/results-fingerprint.png)

### Machine Learning Algorithms ###

We compared many different regression algorithms to get a model able to estimate the user position coordinates based on the RSSI. We trained these algorithms using the medicoes.csv dataset (datasets/medicoes.csv) which we briefly describe previously. The following results show that these models are capable of overcoming the fingerprint naive implementation accuracy. We also tried to combinate different models to overcome issues related outliers but we still a more depth investigation.

	$ python run_ml.py

![machine-leaning](https://github.com/marcelsan/commoveis/blob/master/outputs/results-ml.png)

# References

[1] VO, Quoc Duy; DE, Pradipta. A survey of fingerprint-based outdoor localization. IEEE Communications Surveys & Tutorials, v. 18, n. 1, p. 491-506, 2016. <br />
[2] Pandas. https://pandas.pydata.org <br />
[3]	scikit-lean. http://scikit-learn.org <br />