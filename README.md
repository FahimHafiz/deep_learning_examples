# deep_learning_examples
 Different Basic Deep Learning Problems and Analysis


## Problem1: Diabetes Prediction using a Neural Network

## Project Overview of Problem-1

This project aims to predict diabetes using a neural network implemented with TensorFlow Keras. The project explores various aspects of model training and evaluation, including:

1.  **Plain Validation:** Training and testing the model with different train-test ratios to find the optimal split.
2.  **K-Fold Cross-Validation:** Implementing k-fold cross-validation to assess model performance across different data folds.
3.  **Hyperparameter Tuning:** Investigating the impact of varying:
    *   The number of neurons in the hidden layers.
    *   The activation functions in the hidden layers.
    *   The number of hidden layers.
4.  **Varying Epochs**: Investigating the impact of varying epochs on the model performance.

## Code Structure and Analysis

### 1. Data Loading and Preprocessing (`load_data`, `DataPreprocessor`)

*   The `load_data` function reads the Pima Indians Diabetes dataset from a CSV file ('pima-indians-diabetes.csv'). It separates the dataset into features (X) and target variable (y).
*   The `DataPreprocessor` class provides methods for:
    *   `preprocess`: Standardizes the features using `StandardScaler` from `sklearn.preprocessing`.
    *   `plot_correlation`: Visualizes the correlation between features using a heatmap.
    *   `dataset_summary`: Prints descriptive statistics, missing values, and total null values in the dataset.
    *   `check_class_balance`: Prints the number of samples in each class (0 and 1).

### 2. Model Definition (`DiabetesClassifier`)

*   The `DiabetesClassifier` class defines the neural network model using TensorFlow Keras. It consists of:
    *   An input layer with a shape determined by the number of features.
    *   Two hidden layers with configurable numbers of neurons and activation functions.
    *   An output layer with a sigmoid activation function for binary classification.

### 3. Plain Validation (`main_plain`)

*   This function performs plain validation by splitting the dataset into training and testing sets using `train_test_split`.
*   It trains the model using the training data and evaluates it on the testing data.
*   It calculates and prints accuracy, precision, recall, ROC AUC, and confusion matrix.
*   The function also generates plots of model accuracy and loss over epochs and the confusion matrix.

### 4. K-Fold Cross-Validation (`main_cv`)

*   This function implements k-fold cross-validation using `KFold` from `sklearn.model_selection`.
*   It splits the training data into k folds and iteratively trains the model on k-1 folds and validates on the remaining fold.
*   It calculates average validation metrics (accuracy, precision, recall, ROC AUC) across all folds.
*   It trains a final model on the entire training set and evaluates it on a separate test set.
*   The function also generates a plot of the average validation confusion matrix.

### 5. Hyperparameter Tuning

#### 5.1 Varying Neurons in Layer 1 (`main_nlayer`)

*   This function trains and evaluates the model with different numbers of neurons in the first hidden layer, within a specified range.
*   It uses plain validation with a fixed test size.
*   It collects and prints the results (accuracy, precision, recall, ROC AUC) for each neuron configuration.
*   The function also generates plots of model accuracy and loss over epochs and the confusion matrix for each configuration.

#### 5.2 Varying Neurons in Layer 2 (`main_nlayer`)

*   This function is similar to the previous one but varies the number of neurons in the second hidden layer.

#### 5.3 Varying Activation Function in Layer 1 (`main_actv_fnc`)

*   This function trains and evaluates the model with different activation functions in the first hidden layer.
*   It uses plain validation with a fixed test size.
*   It collects and prints the results (accuracy, precision, recall, ROC AUC) for each activation function.
*   The function also generates plots of model accuracy and loss over epochs and the confusion matrix for each function.

#### 5.4 Varying Activation Function in Layer 2 (`main_actv_fnc`)

*   This function is similar to the previous one but varies the activation function in the second hidden layer.

#### 5.5 Varying Number of Hidden Layers (`main_hid_layer`)

*   This function trains and evaluates the model with different numbers of hidden layers in the range \[1,4].
*   It uses plain validation with a fixed test size.
*   It collects and prints the results (accuracy, precision, recall, ROC AUC) for each configuration.
*   The function also generates plots of model accuracy and loss over epochs and the confusion matrix for each configuration.

#### 5.6 Varying Number of Epochs (`main_epoch`)

*   This function trains and evaluates the model with different numbers of epochs \[50,250].
*   It uses plain validation with a fixed test size.
*   It collects and prints the results (accuracy, precision, recall, ROC AUC) for each configuration.
*   The function also generates plots of model accuracy and loss over epochs and the confusion matrix for each configuration.



## Results and Discussion for Problem 1

# Summary of Performance Trends for Different Hyperparameter Variations for our Proposed Model for the diabetic datasets in Problem 1

| **Hyperparameter**       | **Variation**                       | **Key Observations**                                             | **Optimal Trade-off**                        |
|---------------------------|-------------------------------------|------------------------------------------------------------------|----------------------------------------------|
| **Validation Type**       | Plain validation                   | Higher train accuracy but fluctuating test accuracy              | Cross-validation stabilizes generalization   |
|                           | Cross-validation                   | More stable performance                                          | **Preferred over plain validation**          |
| **Neurons (Layer 1 & 2)** | Increase (10-20)                   | Improves learning, but excessive neurons overfit                | **12-15 neurons work best**                  |
|                           | Increase (5-15)                    | Increases learning capacity                                      | **8-12 neurons work moderate**               |
| **Activation Function (Layer 1 & 2)** | ReLU                    | High performance across all the metrics in both train and test sets | **Preferred Activation Functions**           |
|                           | Sigmoid/Tanh                       | Moderate performance                                             | Use for specific cases                       |
|                           | Softmax                            | Poor performance                                                 | Not suitable for hidden layers               |
| **Hidden Layers**         | Increase (1-4)                     | More layers improve learning but may overfit                    | **2-3 layers for optimal performance and balance learning** |
| **Epochs**                | Increase (50-250)                  | Initially improves accuracy, but overfitting after 175 epochs   | **150-175 epochs optimal**                   |


## Conclusion

According to our analysis, we conclude that hyperparameters in neural networks are definitely a crucial part of the model's performance optimization. The optimal performance while tuning these hyperparameters is not straightforward and is is sometimes intuition-based. When the sample size becomes small, it becomes more difficult to optimize or find a conclusive pattern toward optimizing the performance. In our future work, we can consider the optimal values for each hyperparameters and make a new model architecture to investigate whether the optimal values of all the hyperparameters together work best considering the performance metrics. Furthermore, we can explore such similar small health datasets and create a pipeline that may involve a more structured approach towards finding the optimal values of these hyperparameters, which increases the model's performance and generalizes the dataset pattern.
