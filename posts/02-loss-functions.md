# ⚙️ Loss Function: Easy read for beginners 👀 (Est. 10 mins)

## 🧠 **Understanding Loss Functions**
In a nutshell, loss functions are like the **guiding stars**✨ for machine learning models. Imagine you're trying to teach a computer to recognize cats in pictures. The loss function is like a teacher telling the computer how wrong or right it is with its guesses. It's super important because it helps the computer learn from its mistakes and get better over time. Whether it's figuring out how far off it is with predicting house prices or deciding if an email is spam or not, the right loss function is like having the perfect coach for the computer to improve its skills.

> Definition: A loss function is a crucial component in machine learning algorithms that measures the disparity between the predicted values and the actual target values.

### 🛠️**What's the Objective of Loss Functions?**
The primary goal of a loss function is to quantify the error or deviation of the model's predictions from the true values.

### 🔄**What types of Loss Functions are there? How do we implement them?**
There are various types of loss functions tailored for different tasks, such as Mean Squared Error (MSE) for regression problems and Cross-Entropy Loss for classification tasks.

## ✅**Regression Loss Functions:**

### 1️⃣ **Mean Squared Error (MSE):** 
Measures the average squared difference between predicted and actual values.
Mathematically, MSE is calculated as:

MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$
- n is the number of data points
- y_i is the actual target value
- ŷ_i is the predicted value for the i-th data point

Example Code
```
import numpy as np
def mean_squared_error(y_true, y_pred):

   diff = y_pred- y_true
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   
   return mean_diff

y_true= np.array([1.1,2,1.7])
y_pred= np.array([1,1.7,1.5])

print(mean_squared_error(y_true,y_pred))
```


OR using **mean_squared_error from sklearn**

```
from sklearn.metrics import mean_squared_error
y_true= np.array([1.1,2,1.7])
y_pred= np.array([1,1.7,1.5])
mean_squared_error(y_true, y_pred)
```

### 2️⃣ **Mean Absolute Error (MAE):**
Computes the average absolute difference between predicted and actual values.
MAE is expressed as:

MAE = $\frac{1}{n} \Sigma_{i=1}^n|{y}-\hat{y}|$
- ∣⋅∣ denotes the absolute value

```
import numpy as np
def mean_absolute_error(y_true, y_pred):
    diff = y_pred - y_true
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

y_true= np.array([1.1,2,1.7])
y_pred= np.array([1,1.7,1.5])
mean_absolute_error(y_true,y_pred)
```

OR using **mean_absolute_error from sklearn**

```
from sklearn.metrics import mean_absolute_error
y_true= np.array([1.1,2,1.7])
y_pred= np.array([1,1.7,1.5])
mean_absolute_error(y_true, y_pred)
```

## ✅ **Classification Loss Functions:**

### 1️⃣ **Binary Cross-Entropy Loss / Log Loss:** 
Appropriate for binary classification problems, penalizing deviations from true binary labels.
The binary cross-entropy loss is defined as:
BCE = - (1/n) * Σ[y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i)]
where
- y_i is the true binary label (0 or 1)
- ŷ_i is the predicted probability of the positive classes

```
import numpy as np

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Small constant to avoid log(0)

    # Ensure y_pred is within the range (epsilon, 1 - epsilon)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate binary cross-entropy loss
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Take the mean over all examples
    mean_loss = np.mean(loss)

    return mean_loss

# Example usage:
true_labels = np.array([1, 0, 1, 1, 0])
predicted_probs = np.array([0.9, 0.2, 0.8, 0.95, 0.1])

bce_loss = binary_cross_entropy_loss(true_labels, predicted_probs)
print(f"Binary Cross-Entropy Loss: {bce_loss:.4f}")
```
OR **Binary Cross-Entropy Loss** can be used by importing **tensorflow**
```
import tensorflow as tf

# Example usage for Binary Cross-Entropy Loss:
true_labels_binary = tf.constant([1, 0, 1, 1, 0], dtype=tf.float32)
predicted_probs_binary = tf.constant([0.9, 0.2, 0.8, 0.95, 0.1], dtype=tf.float32)

bce_loss_binary = tf.keras.losses.BinaryCrossentropy()(true_labels_binary, predicted_probs_binary)
print(f"Binary Cross-Entropy Loss: {bce_loss_binary.numpy():.4f}")
```

### 2️⃣ **Categorical Cross-Entropy Loss:** 
Suitable for multi-class classification, penalizing deviations from true categorical labels.
For multi-class classification, the categorical cross-entropy loss is given by:
CCE = - (1/n) * Σ Σ[y_ij * log(ŷ_ij)]
where
- k is the number of classes
- y_ij is the indicator function (1 if the true class is j, 0 otherwise)
- ŷ_ij is the predicted probability for class j

```
import numpy as np

def categorical_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Small constant to avoid log(0)

    # Ensure y_pred is within the range (epsilon, 1 - epsilon)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate categorical cross-entropy loss
    loss = - np.sum(y_true * np.log(y_pred))

    # Take the mean over all examples
    mean_loss = loss / len(y_true)

    return mean_loss

# Example usage:
true_labels = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # Example one-hot encoded true labels
predicted_probs = np.array([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])

cce_loss = categorical_cross_entropy_loss(true_labels, predicted_probs)
print(f"Categorical Cross-Entropy Loss: {cce_loss:.4f}")
```

OR **Categorical Cross-Entropy Loss** can be used by importing **tensorflow**

```
import tensorflow as tf

# Example usage for Categorical Cross-Entropy Loss:
true_labels_categorical = tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=tf.float32)
predicted_probs_categorical = tf.constant([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype=tf.float32)

cce_loss_categorical = tf.keras.losses.CategoricalCrossentropy()(true_labels_categorical, predicted_probs_categorical)
print(f"Categorical Cross-Entropy Loss: {cce_loss_categorical.numpy():.4f}")
```

### 🧐 **Other Custom Loss Functions:**
In some cases, data scientists may design custom loss functions tailored to specific project requirements. Some examples are, Huber Loss for Robust Regression, Quantile Loss for Quantile Regression, Focal Loss for Imbalanced Classification, and Dice Loss for Image Segmentation.

## 🎯 **So How do you choose which Loss Function?**
The choice of a loss function depends on the nature of the problem and the desired characteristics of the model. For instance, robustness to outliers may lead to the selection of a different loss function.

## 🤔 **Evaluation Metric vs Loss Function:**
While loss functions guide the model during training, evaluation metrics (accuracy, precision, recall, etc.) assess the model's performance on unseen data.

## 🔄 **Trade-offs:**
There are trade-offs between different loss functions, and selecting an appropriate one involves considering the specific challenges and objectives of the machine learning task.

## 📢 **Conclusion:**
So, why does this matter? Well, imagine you're training a robot to pick up objects without dropping them. If the loss function isn't good, the robot might keep making mistakes, dropping things all over the place. But with the right loss function, it's like giving the robot a clear scorecard to learn from – helping it become a master at picking up things without a hitch. So, loss functions are like the secret sauce that helps computers and robots become smarter and better at the tasks we want them to do!