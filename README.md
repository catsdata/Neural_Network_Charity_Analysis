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

Beks has come a long way since her first day at that boot camp five years ago—and since earlier this week, when she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special consideration for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively


## Resources

- Data Sources: 
    - [Charity Data](https://github.com/catsdata/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)
- Software/Packages:  
    - Machine Learning (mlenv)
        - Python
        - Pandas
        - SKLearn
            - StandardScaler
            - MinMaxScaler
            - PCA
            - KMeans
    - Jupyter Notebook

## Results

### Preprocessing Data for a Neural Network Model

Using your knowledge of Pandas and the Scikit-Learn’s StandardScaler(), you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2.

1. Open the AlphabetSoupCharity_starter_code.ipynb file, rename it AlphabetSoupCharity.ipynb, and save it to your Neural_Network_Charity_Analysis GitHub folder.
2. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
    - What variable(s) are considered the target(s) for your model?
    - What variable(s) are considered the feature(s) for your model?
3. Drop the EIN and NAME columns.
4. Determine the number of unique values for each column.
5. For those columns that have more than 10 unique values, determine the number of data points for each unique value.
6. Create a density plot to determine the distribution of the column values.
7. Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, Other, and then check if the binning was successful.
8. Generate a list of categorical variables.
9. Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame.
10. Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals.

At this point, your merged DataFrame should look like this: [link](https://courses.bootcampspot.com/courses/913/files/1379957/preview)

The merged DataFame shows four columns: STATUS, ASK_AMT, IS_SUCCESFUL, APPLICATION_TYPE_Other,  APPLICATION_TYPE_T_10, APPLICATION_TYPE_T19, APPLICATION_TYPE_T3 and contains 5 rows and 44 columns.

11. Split the preprocessed data into features and target arrays.
12. Split the preprocessed data into training and testing datasets.
13. Standardize numerical variables using Scikit-Learn’s StandardScaler class, then scale the data.



### Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
2. Create the first hidden layer and choose an appropriate activation function.
3. If necessary, add a second hidden layer with an appropriate activation function.
4. Create an output layer with an appropriate activation function.
5. Check the structure of the model.
6. Compile and train the model.
7. Create a callback that saves the model's weights every 5 epochs.
8. Evaluate the model using the test data to determine the loss and accuracy.
9. Save and export your results to an HDF5 file, and name it AlphabetSoupCharity.h5.

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

