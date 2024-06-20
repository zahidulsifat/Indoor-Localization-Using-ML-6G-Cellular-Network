# This model was developed in Python using Pandas and scikit-learn libraries.

## Proposed System
![Proposed System](https://github.com/zahidulsifat/Indoor-Localization-6G-Cellular-Network/blob/main/Outputs/Proposed%20System.png)


Here, in this figure, a traditional method called signal fingerprinting is applied, consisting of two phases: training and matching. The matching algorithm is used during the matching phase. The system demonstrates the sequence of tasks involved in generating and matching fingerprints and the process of populating the fingerprint database with geotagged signatures. In order to develop a model that can estimate the user position coordinates based on the Received Signal Strength (RSSI), I have compared a wide range of regression algorithms. 


# Data Collection Processing and Visualization

Data preprocessing included shuffling and normalization of attributes to prepare the dataset for algorithmic evaluation. The data used in this study contains about 2557 lines with six different measures from six different Radio Base Stations (RBSs). It has two columns that represent the user's position. Also, two additional datasets are used: one provides more information about the RBSs, such as their positions and the power of the radiated signal, and another is used as a test dataset.

## Heatmap showing the correlation between user positions and RSSI values from training dataset
![Correlation Matrix](https://github.com/zahidulsifat/Indoor-Localization-Using-ML-6G-Cellular-Network/blob/main/Visualization/Correlation%20Matrix%20of%20RSSI.png)

## Scatter plot of correlation between the user's position and the RSSI values from all Radio Base Station from training datase
![Scatter Plot](https://github.com/zahidulsifat/Indoor-Localization-Using-ML-6G-Cellular-Network/blob/main/Visualization/Scatter%20Plot%20For%20All%20RBS.png)


# Results

## Algorithmic Performance Comparison
![Regration Algorithm](https://github.com/zahidulsifat/Indoor-Localization-6G-Cellular-Network/blob/main/Outputs/Regression%20Algorithm.png)

## Evaluation of K Nearest Neighbor (KNN) (k=3) Model

### Distance Error Histogram

![Distance Error Histogram](https://github.com/zahidulsifat/Indoor-Localization-6G-Cellular-Network/blob/main/Outputs/Distribution%20of%20Geodesic%20Errors.png)

### Actual vs. Predicted Distance

![Actual vs. Predicted Distance](https://github.com/zahidulsifat/Indoor-Localization-6G-Cellular-Network/blob/main/Outputs/Actual%20Vs%20Predicted%20Distance%20Errors.png)


## Cross-Validation
 The algorithm uses Leave-One-Out cross-validation to assess the model's accuracy. This cross-validation filters out poorly predicted points and refines the model accordingly.


# For Further Development

[1] Open this folder as project folder in any Python IDE <br />
[2] The above mentioned folder has datasates included <br />
[3] Here Python 3.10 was used as based interpreter <br />
[4] To perform simulation, first, create a virtual environment in a blank folder then install the required libraries <br />
[5] Pycharm Professional Edition is Highly Recommended <br />


## File Information 

### Accuracy_Percentage
For calculating percentage result from mean error 

### Fingerprint_Algorithm
Fingerprint algorithm used for localization

### KNN_Evaluation
For evaluating the performance of K Nearest Neighbor (KNN) Model

### Leaveoneout_Cross_Validation
This cross-validation filters out poorly predicted points and refines the model accordingly

### LTE_ML_Model
The new model is based on this model

### Revised_6G_ML_Model
The new model for localization in 6G indoor environment

### Test_Model
This was the initial model

### Traditiuonal_Models
Traditional models like Sui, OkamuraHata, Ericsson are included in this.

### Tranditional_Models_With_ML
Traditional models with machine learning algorithms


# References

**[1]** M. Z. Islam Sifat, M. Mostafa Amir Faisal, M. M. Hossain and M. A. Islam, "An Improved Positioning System For 6G Cellular Network," 2023 26th International 
        Conference on Computer and Information Technology (ICCIT), Cox's Bazar, Bangladesh, 2023, pp. 1-6, doi: https://10.1109/ICCIT60459.2023.10441118 <br />
**[2]** VO, Quoc Duy; DE, Pradipta. A survey of fingerprint-based outdoor localization. IEEE Communications Surveys & Tutorials, v. 18, n. 1, p. 491-506, 2016, doi: 
        https://doi.org/10.1109/COMST.2015.2448632 <br />
**[3]** Pandas. https://pandas.pydata.org <br />
**[4]**	scikit-lean. http://scikit-learn.org <br />
