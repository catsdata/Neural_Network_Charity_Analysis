# Neural Network Charity Analysis

**Table of Contents**

1. [Overview](https://github.com/catsdata/Neural_Network_Charity_Analysis#overview)
2. [Resources](https://github.com/catsdata/Neural_Network_Charity_Analysis#resources)
3. [Results](https://github.com/catsdata/Neural_Network_Charity_Analysis#results)
    - [Preprocessing Data for a Neural Network Model](https://github.com/catsdata/Neural_Network_Charity_Analysis#preprocessing-data-for-a-neural-network-model)
    - [Compile, Train, and Evaluate the Model](https://github.com/catsdata/Neural_Network_Charity_Analysis#compile-train-and-evaluate-the-model)
    - [Optimize the Model](https://github.com/catsdata/Neural_Network_Charity_Analysis#optimize-the-model)
4. [Summary](https://github.com/catsdata/Neural_Network_Charity_Analysis#summary)


## Overview

Using a data set of 34,000 organizations that have been funded over the years, create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

The following detail has been provided:

- EIN and NAME:  Identification columns
- APPLICATION_TYPE:  Alphabet Soup application type
- AFFILIATION:  Affiliated sector of industry
- CLASSIFICATION:  Government organization classification
- USE_CASE:  Use case for funding
- ORGANIZATION:  Organization type
- STATUS:  Active status
- INCOME_AMT:  Income classification
- SPECIAL_CONSIDERATIONS:  Special consideration for application
- ASK_AMT:  Funding amount requested
- IS_SUCCESSFUL:  Was the money used effectively


## Resources

- Data Sources: 
    - [Charity Data](https://github.com/catsdata/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)
- Software/Packages:  
    - Machine Learning (mlenv)
        - Python
        - Pandas
        - SKLearn
            - StandardScaler
            - OneHotEncoder
            - MinMaxScaler
            - PCA
            - KMeans
        - TensorFlow
    - Jupyter Notebook

## Results

### Preprocessing Data for a Neural Network Model

Using Pandas and the Scikit-Learn’s StandardScaler(), the dataset was preprocessed the dataset in order to compile, train, and evaluate the neural network model.

- After determining denisty of unique values; CLASSIFICATION and APPLICATION_TYPE were binned to reduce variables
- Encoded categorical variables using one-hot encoding into a new DataFrame & merged with the original DataFrame
- Split the data into Target and Features arrays for training and testing datasets
    - Target set to "Is_Successful" array 
    - Features set to:
        - APPLICATION_TYPE
        - AFFILIATION
        - CLASSIFICATION
        - USE_CASE
        - ORGANIZATION
        - INCOME_AMT
        - SPECIAL_CONSIDERATIONS
- Fit and Scaled the data using StandardScaler()

![del1](https://github.com/catsdata/Neural_Network_Charity_Analysis/blob/main/images/del1.PNG)

### Compile, Train, and Evaluate the Model

Using TensorFlow, designed a neural network to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. 

- Created a neural network model by assigning the count of input features for each layer using Tensorflow Keras
- Created two hidden layers with "relu" activation with 10 and 5 nodes respecitively
- Created an output layer with a "sigmoid" activation
- Compiled and trained the model with a callback that saves the model's weights every 5 epochs.
- Evaluated the model's loss 55.2% and accuracy 72.6%.
- Saved and exported results to an HDF5 file: [AlphabetSoupCharity.h5](https://github.com/catsdata/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.h5).

![del2](https://github.com/catsdata/Neural_Network_Charity_Analysis/blob/main/images/del2.PNG)

### Optimize the Model

Attempted to optimize the model in order to achieve a target predictive accuracy higher than 75%. 

#### Attempt #1

I started simple and added an additional layer and spread out the nodes to 20, 10, and 5 respectively.  Activation remained the same, but parameters increased from 501 to 1,151.

![opt1](https://github.com/catsdata/Neural_Network_Charity_Analysis/blob/main/images/opt1.PNG)

The loss increased from 55.2% to 55.5% and accuracy remained 72.6%.   So these changes made very little impact to the results.

[Optimization1.ipynb]()

#### Attempt #2

I decided to look into the data types of the data a little closer.  The ASK_AMT was an integer and the model is built for object datatypes.  So I converted it and looked at the data points.  $5000 was the primary requested amount, so I binned as $5000 vs Other.  I left the edits to the layers and nodes alone from the first attempt since it didn't impact the results, but the additon of ASK_AMT to the test increased the total parameters to 1,171.

![opt2](https://github.com/catsdata/Neural_Network_Charity_Analysis/blob/main/images/opt2.PNG)

The loss decreased from 55.2% to 55.1% and accuracy increased slightly to 72.8%.   So again, these changes made very little impact to the results.

[Optimization2.ipynb]()

#### Attempt #3

I decided to scrap the idea of the ASK_AMT being a factor of data that would help.  So this time I went the route of dropping additional features.  Along with EIN and NAME, I also dropped STATUS and SPECIAL_CONSIDERATIONS as the data spread on them were insignificant.  I also decided to try the sigmoid activation on all layers, and increased the nodes to 100, 50, and 25.  Parameters were now substantial at 10,451.

![opt3](https://github.com/catsdata/Neural_Network_Charity_Analysis/blob/main/images/opt3.PNG)

The loss increased to 55.5% and accuracy decreased to 72.6%.  And again, these changes made very little impact to the results.

[Optimization3.ipynb]()

#### Attempt #4

When playing with TensorFlow Playgroungm I recall that some datasets were a bit complicated and even though no changes seemed to occur within 100-200 epochs, that the good results came in closer to 1000 epochs.  So perhaps I'm just impatient.  I chose to modify my attempt from original code (not using ASK_AMT binning), with the original relu activation on 3 hidden layers, but with 150, 75, and 25 nodes.  Total parameters 19,581.  I increased the epochs to 1000, and reduced the callback checkpoint to every 25 epochs to reduce activity a little.  This one took quite a while to run.  I played at least 20 hands of solitaire.  

![opt4](https://github.com/catsdata/Neural_Network_Charity_Analysis/blob/main/images/opt4.PNG)

The loss increased substantially to 75.5% and accuracy decreased a teeny bit to 72.5%.  It's here that I gave up.  Seems everything was hanging at that same accuracy point.

[AlphabetSoupCharity_Optimization.ipynb]()

## Summary 

Overall, I was unsuccessful in improving the model from the original.  Although there may be a solution that will increase the accuracy to over 75%, perhaps Neural/Deep learning isn't the way to go.  A suggestion would be to use Random Forest machine learning to investigate possible clustering of data.  They are built to handle outliers in data, as well as data being non-linear, which we may have in this case.  
