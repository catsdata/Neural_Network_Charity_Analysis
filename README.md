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

Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

NOTE
The accuracy for the solution is designed to be lower than 75% after completing the requirements for Deliverables 1 and 2.

Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

- Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
    - Dropping more or fewer columns.
    - Creating more bins for rare occurrences in columns.
    - Increasing or decreasing the number of values for each bin.
- Adding more neurons to a hidden layer.
- Adding more hidden layers.
- Using different activation functions for the hidden layers.
- Adding or reducing the number of epochs to the training regimen.

NOTE
You will not lose points if your model does not achieve target performance, as long as you make three attempts at optimizing the model in your AlphabetSoupCharity_Optimzation.ipynb file.

1. Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
2. Import your dependencies, and read in the charity_data.csv to a Pandas DataFrame.
3. Preprocess the dataset like you did in Deliverable 1, taking into account any modifications to optimize the model.
4. Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
5. Create a callback that saves the model's weights every 5 epochs.
6. Save and export your results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5.
7. Save your AlphabetSoupCharity_Optimzation.ipynb file and AlphabetSoupCharity_Optimization.h5 file to your Neural_Network_Charity_Analysis folder.

## Summary 

### Results: 
Using bulleted lists and images to support your answers, address the following questions.

Data Preprocessing
- What variable(s) are considered the target(s) for your model?
- What variable(s) are considered to be the features for your model?
- What variable(s) are neither targets nor features, and should be removed from the input data?

Compiling, Training, and Evaluating the Model
- How many neurons, layers, and activation functions did you select for your neural network model, and why?
- Were you able to achieve the target model performance?
- What steps did you take to try and increase model performance?

### Summary: 
Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.

