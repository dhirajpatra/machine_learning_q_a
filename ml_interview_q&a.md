1. **What is *Bias* in Machine Learning?**  
   	Bias in machine learning refers to the tendency of a machine learning model to consistently make errors in a particular direction, based on the data it has been trained on. This can happen when the training data used to train the model is not representative of the real-world population that the model will be used on.  
     
   For example, if a machine learning model is trained on a dataset that contains more data for a particular class or demographic, the model may become biased towards that class or demographic, resulting in poor performance for other classes or demographics. This is known as "dataset bias".  
     
   Another type of bias in machine learning is "algorithmic bias", which occurs when the algorithms used in the model inherently favor certain classes or demographics. This can happen when the model is designed with specific assumptions or features that may not apply to all classes or demographics equally.  
     
   Bias in machine learning can have serious consequences, especially when the model is used in critical decision-making processes such as hiring, lending, or medical diagnosis. Therefore, it is important to identify and address bias in machine learning models to ensure fair and accurate results.

2. **What is the *Bias-Variance* tradeoff?** 

The bias-variance tradeoff is a fundamental concept in machine learning that refers to the relationship between the bias and variance of a model and its predictive performance.

Bias refers to the error that is introduced by approximating a real-world problem with a simpler model. A model with high bias will underfit the data and may not capture the complexity of the underlying relationship between the input features and the target variable. In other words, it is too simple and doesn't capture all the important features of the problem.

Variance, on the other hand, refers to the sensitivity of the model to changes in the training data. A model with high variance will overfit the data and may capture noise or random fluctuations in the training data, leading to poor performance on new, unseen data.

The bias-variance tradeoff occurs because decreasing one typically leads to an increase in the other. The goal in machine learning is to find the sweet spot between bias and variance that results in the best predictive performance on new, unseen data.

The tradeoff can be visualized as a U-shaped curve, with the total error of the model on the y-axis and the complexity of the model on the x-axis. As the complexity of the model increases, the bias decreases and the variance increases. The total error is minimized at the point where the sum of the bias and variance is the lowest.

3. **How can you identify a *High Bias model*? How can you fix it?**

A high bias model is a model that is too simple and has underfit the data. This can be identified by comparing the training error and the validation error of the model. If both errors are high, then the model is likely underfitting the data.

To fix a high bias model, you can take the following steps:

Increase the complexity of the model: This can be done by adding more features to the model or increasing the number of hidden layers in a neural network. By doing this, you allow the model to capture more complex relationships between the input and output variables.

Reduce regularization: Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function. However, if the model has high bias, it may be too restrictive. By reducing the regularization, the model can become less biased.

Change the algorithm: Some algorithms may not be suitable for certain types of problems. For example, linear regression may not be suitable for non-linear problems. By choosing a more appropriate algorithm, you may be able to reduce bias.

Increase the amount of training data: A model may have high bias if it hasn't been trained on enough data. By increasing the amount of training data, the model can learn more complex relationships between the input and output variables.

Change the model architecture: If the model architecture is too simple, it may be necessary to change it. This can involve adding more layers, changing the activation function, or using a different type of model altogether.

4. **Provide an intuitive explanation of the *Bias-Variance Tradeoff***  

The bias-variance tradeoff is a concept in machine learning that refers to the relationship between the complexity of a model and its ability to generalize to new, unseen data.

Imagine you are teaching a group of students how to throw a basketball through a hoop. You start by teaching them the basics: how to hold the ball, how to aim, and how to release the ball. As they practice, you notice that some students are consistently throwing the ball too high or too low, while others are missing the hoop entirely.

The students who are throwing the ball too high or too low are underfitting the task \- their throws are biased towards a certain direction. They need more guidance and practice to refine their technique and reduce their bias. On the other hand, the students who are missing the hoop entirely are overfitting the task \- their throws are too sensitive to minor variations in the environment, such as wind or the position of the hoop. They need to simplify their approach and focus on the essential elements of the task to reduce their variance.

In machine learning, a high-bias model is one that is too simple and has underfit the data, while a high-variance model is one that is too complex and has overfit the data. The goal is to find the right balance between bias and variance that results in the best performance on new, unseen data. This is similar to finding the right balance between teaching the students too much or too little \- you want to provide them with enough guidance and practice to improve their technique and accuracy, but not so much that they become overly sensitive to minor variations in the environment.

5. **Name some types of Data Biases in Machine Learning?**  

**Here are some types of data biases that can occur in machine learning:**

Selection bias: This occurs when the data used to train a model is not representative of the population it is intended to be used on. This can happen when the data is collected from a biased sample or when certain groups are underrepresented in the data.

Measurement bias: This occurs when the way data is collected or measured introduces systematic errors or inaccuracies. For example, if a sensor is biased and consistently reports values that are too high or too low, this can lead to measurement bias.

Confirmation bias: This occurs when the data is selected or interpreted in a way that confirms pre-existing beliefs or hypotheses, rather than being analyzed objectively.

Sampling bias: This occurs when the data sample is not randomly selected and does not accurately reflect the underlying distribution of the population.

Survivorship bias: This occurs when only successful or surviving instances are included in the data, leading to an incomplete or biased understanding of the phenomenon being studied.

Prejudice bias: This occurs when the data or the algorithm reflects the prejudice or stereotypes of its creators or society as a whole. For example, if the data is biased against a particular race or gender, the algorithm trained on that data may also exhibit bias.

6. **What's the difference between Bagging and Boosting algorithms?  Related To: Ensemble Learning, Data Processing, Classification**

Bagging and Boosting are both techniques used in ensemble learning, which involves combining multiple machine learning models to improve their accuracy and overall performance.

Bagging, short for Bootstrap Aggregating, is a technique that involves creating multiple bootstrapped samples of the original dataset and training each model on a different sample. These models are then combined through a voting process, where each model's predictions are given equal weight. Bagging is commonly used in decision tree-based algorithms, such as Random Forest, and can help reduce overfitting by creating a more diverse set of models.

Boosting, on the other hand, is a technique that involves training a sequence of models, with each subsequent model attempting to correct the errors made by the previous model. Unlike bagging, boosting assigns weights to each training example, so that the subsequent models give more emphasis to the examples that were not correctly predicted in the previous iteration. Boosting can improve the accuracy of the model, but it's more susceptible to overfitting and can be slower to train than bagging.

In summary, while both bagging and boosting are ensemble learning techniques used for data processing and classification tasks, they differ in their approach. Bagging focuses on creating diverse models from different bootstrapped samples of the dataset and combining them through a voting process. Boosting, on the other hand, trains a sequence of models, where each model attempts to correct the errors of the previous model by assigning weights to the training examples.

7. **What is the Bias Error?  Related To: Supervised Learning**

In supervised learning, bias error refers to the error that occurs when a machine learning model is unable to capture the true relationship between the input features and the target variable. Bias error can occur when a model is too simple to capture the complexity of the data, resulting in underfitting.

A model with high bias error may have low accuracy on both the training and test datasets. It is important to reduce bias error to create a model that can accurately predict the target variable.

Bias error can be reduced by increasing the complexity of the model. This can be done by adding more features, increasing the number of hidden layers in a neural network, or choosing a more complex model architecture. However, increasing the complexity of the model also increases the risk of overfitting, where the model becomes too specialized to the training data and performs poorly on new, unseen data.

In summary, bias error is a type of error that occurs in supervised learning when a model is too simple to capture the true relationship between the input features and the target variable. It can be reduced by increasing the complexity of the model, but this should be done with caution to avoid overfitting.

8. **What is the Variance Error?  Related To: Supervised Learning**

In supervised learning, variance error refers to the error that occurs when a machine learning model is overly complex and captures the noise or randomness in the training data instead of the true underlying pattern. Variance error can occur when a model is overfitting to the training data.

A model with high variance error may have high accuracy on the training dataset, but poor accuracy on the test dataset or new, unseen data. This is because the model has become too specialized to the training data and is unable to generalize to new data.

To reduce variance error, it is important to simplify the model and reduce its complexity. This can be done by removing features that are not important or by using regularization techniques, such as L1 or L2 regularization, that penalize the model for having large weights. Cross-validation can also be used to help identify the optimal level of model complexity.

9. **How can you relate the KNN Algorithm to the Bias-Variance tradeoff?  Related To: K-Nearest Neighbors**

KNN algorithm can be related to the bias-variance tradeoff as a low-bias, high-variance algorithm that is prone to overfitting when k is too low and can become more biased when k is too high. The optimal value of k can be selected through techniques such as cross-validation to achieve the optimal balance between bias and variance.

10. **What to do if you have High Variance Problem?**  

If you have a high variance problem in a machine learning model, it means that the model is overfitting to the training data and is unable to generalize well to new, unseen data. To address this issue, you can try the following approaches:

Reduce the complexity of the model: One way to reduce variance is to simplify the model by removing unnecessary features, reducing the number of layers in a neural network, or reducing the number of trees in a random forest. This can help the model focus on the most important features and reduce its tendency to overfit.

Use regularization: Regularization is a technique that adds a penalty term to the model's loss function to prevent it from overfitting. There are different types of regularization techniques such as L1 and L2 regularization that can be used to reduce variance.

Increase the size of the training set: Having more data can help the model better capture the underlying patterns and reduce the impact of noise in the data.

Use data augmentation: Data augmentation techniques such as rotation, flipping, and zooming can help increase the size of the training set and make the model more robust to variations in the data.

Ensemble learning: Ensemble learning involves combining the predictions of multiple models to make a final prediction. This can help reduce variance by reducing the impact of individual model's errors.

Cross-validation: Cross-validation can help identify the optimal level of model complexity and the best hyperparameters for the model. This can help prevent overfitting and improve the generalization performance of the model.

11. **What to do if you have High Bias Problem?**  

If you have a high bias problem in a machine learning model, it means that the model is underfitting and is unable to capture the underlying patterns in the data. To address this issue, you can try the following approaches:

Increase the complexity of the model: One way to reduce bias is to increase the complexity of the model by adding more layers to a neural network, increasing the number of trees in a random forest, or adding more features to the model. This can help the model capture more complex relationships between the input features and the target variable.

Use a different model: If the current model is too simple to capture the underlying patterns in the data, you can try using a more complex model such as a deep neural network, a support vector machine (SVM), or a gradient boosting machine (GBM).

Feature engineering: Feature engineering involves creating new features from the existing ones to help the model better capture the underlying patterns in the data. This can involve techniques such as scaling, transforming, or combining the features.

Reduce regularization: Regularization is a technique used to reduce variance, but it can also increase bias. If the model is too biased, you can try reducing the amount of regularization or using a different type of regularization.

Increase the size of the training set: Having more data can help the model better capture the underlying patterns and reduce bias.

Cross-validation: Cross-validation can help identify the optimal level of model complexity and the best hyperparameters for the model. This can help prevent underfitting and improve the generalization performance of the model.

12. **When you sample, what potential Sampling Biases could you be inflicting?  Related To: Statistics, Data Processing**

When you sample data from a population, there are several potential sampling biases that you could be inflicting. Some of the most common types of sampling biases are:

Selection bias: This occurs when the sample is not representative of the population, which can happen if certain groups are overrepresented or underrepresented in the sample. For example, if you are conducting a survey on income levels and only survey people in a wealthy neighborhood, you may overestimate the income levels of the population.

Survivorship bias: This occurs when you only consider the data that is available, which can lead to a skewed view of the population. For example, if you are analyzing stock market returns and only look at the successful companies, you may miss the companies that failed.

Sampling frame bias: This occurs when the sampling frame (i.e., the list of individuals or items from which the sample is drawn) is not representative of the population. For example, if you are conducting a survey of college students and only use a list of students from a specific college, you may miss students from other colleges.

Non-response bias: This occurs when individuals who are selected to participate in the sample do not respond or participate in the study. This can lead to a biased sample if the non-responders differ systematically from the responders.

Measurement bias: This occurs when the measurement instrument or the data collection method produces systematic errors. For example, if you are using a self-reported survey to measure weight, some people may underreport their weight.

13. **How to identify a High Variance model? How do you fix it?**  

Identifying a high variance model is an important step in improving the accuracy of a machine learning model. A high variance model occurs when the model is overly complex and fits the training data too closely, resulting in poor generalization performance on new, unseen data. Here are some ways to identify a high variance model:

High training accuracy, but low validation accuracy: If the model has high accuracy on the training set but performs poorly on the validation set, it is likely to be overfitting and has high variance.

Large difference between training and validation accuracy: If the difference between the training accuracy and validation accuracy is significant, it is an indication that the model has high variance.

High sensitivity to small changes in the training data: If the model's performance changes significantly with minor variations in the training data, it is likely to have high variance.

To fix a high variance model, you can try the following approaches:

Reduce model complexity: High variance can be reduced by simplifying the model. This can be done by reducing the number of features, decreasing the model depth, or decreasing the regularization parameters.

Increase the training set size: Increasing the amount of training data can help the model better capture the underlying patterns in the data and reduce overfitting.

Regularization: Regularization techniques such as L1 or L2 regularization can be used to reduce overfitting and improve the generalization performance of the model.

Data augmentation: Data augmentation techniques such as rotating, scaling, or flipping the images can be used to increase the size of the training set and reduce overfitting.

Ensemble learning: Using ensemble learning techniques such as bagging or boosting can help reduce variance by combining multiple models to make predictions.

14. **What are the differences between Content-Based and Collaborative Methods in terms of Bias and Variance?  Related To: Recommendation Systems**

Content-Based and Collaborative Filtering are two popular methods used in recommendation systems. They differ in their approach to recommending items to users and have different biases and variances.

Content-Based Filtering:

Content-Based Filtering is a recommendation system that recommends items to users based on their past behavior and the attributes of the items. The recommendation system looks at the user's profile and recommends items that are similar to the items that the user has previously interacted with. Content-Based Filtering tends to have a high bias and low variance.

Bias: Content-Based Filtering has a high bias because it only recommends items that are similar to the items that the user has previously interacted with. This can result in a narrow range of recommendations, and users may not discover new items that they may be interested in.

Variance: Content-Based Filtering has low variance because it relies on the attributes of the items and the user's past behavior, and not on other users' opinions. Therefore, the recommendations are less susceptible to fluctuations in other users' preferences.

Collaborative Filtering:

Collaborative Filtering is a recommendation system that recommends items to users based on the behavior of similar users. It does not rely on the attributes of the items but rather on the interactions between users and items. Collaborative Filtering tends to have a low bias and high variance.

Bias: Collaborative Filtering has a low bias because it can recommend items that the user has not previously interacted with, based on the behavior of similar users. This can lead to a broader range of recommendations.

Variance: Collaborative Filtering has high variance because it is based on the behavior of other users. It can be sensitive to fluctuations in other users' preferences and may recommend items that are not relevant to the user's interests.

15. **What is a Perceptron?  Related To: Supervised Learning, Neural Networks**

A Perceptron is a type of neural network that is used for binary classification tasks. It is a single-layer neural network that consists of input units, a processing unit, and an output unit. The input units receive the input data, and the processing unit computes the weighted sum of the inputs. The output unit then applies a threshold function to the weighted sum to generate a binary output.

During training, the weights of the inputs are adjusted iteratively using a learning rule, such as the perceptron learning rule or the delta rule, to minimize the error between the predicted output and the actual output. The learning rule updates the weights based on the difference between the predicted output and the target output, and this process continues until the error is minimized.

Perceptrons were first introduced by Frank Rosenblatt in the late 1950s and were used extensively for pattern recognition and classification tasks. While perceptrons are simple and efficient, they are limited to linearly separable problems and cannot handle complex nonlinear relationships. However, they serve as a foundational building block for more complex neural networks such as multi-layer perceptrons (MLPs) and convolutional neural networks (CNNs).

16. **Why Naive Bayes is called Naive?  Related To: Naïve Bayes, Supervised Learning**

Naive Bayes is called "naive" because it makes a simplifying assumption that is often not true in real-world applications. Specifically, it assumes that the features in a dataset are independent of each other, which is a strong and often unrealistic assumption.

This assumption simplifies the calculation of probabilities in the model, making it computationally efficient and easy to implement. However, it can lead to inaccurate predictions when the features in a dataset are correlated or dependent on each other. Despite this limitation, Naive Bayes has been shown to work well in many practical applications, especially in text classification and spam filtering, where the assumption of independence among features is often not too far from reality.

Despite its "naive" assumption, Naive Bayes is a powerful and widely used algorithm in supervised learning. It works by calculating the conditional probability of each class given the features in the input data and selecting the class with the highest probability as the predicted output. It can handle both categorical and continuous input data and has been shown to perform well in many real-world scenarios, especially with high-dimensional data.

17. **What is a Decision Boundary?  Related To: Logistic Regression**

In logistic regression and other classification algorithms, a decision boundary is a boundary that separates the input space into different regions, each corresponding to a different class label. The decision boundary is determined by the model parameters and the type of algorithm used.

For example, in a binary classification problem with two features (such as age and income), the decision boundary is a line that separates the input space into two regions, one for each class label. Instances on one side of the line are classified as belonging to one class, while instances on the other side of the line are classified as belonging to the other class.

In logistic regression, the decision boundary is a line (in the case of two features) or a hyperplane (in the case of more than two features) that separates the input space into two regions. The goal of the algorithm is to find the optimal decision boundary that maximizes the likelihood of the training data.

The decision boundary is a critical aspect of classification algorithms, as it determines how the algorithm will classify new instances. The location and shape of the decision boundary can have a significant impact on the performance of the algorithm, and different algorithms may produce different decision boundaries for the same dataset.

18. **What is the difference between KNN and K-means Clustering?  Related To: K-Means Clustering, Supervised Learning, Unsupervised Learning**

KNN (k-nearest neighbors) and K-means clustering are two different algorithms used in machine learning, and they differ in several ways:

Supervised vs unsupervised learning: KNN is a supervised learning algorithm, while K-means clustering is an unsupervised learning algorithm. This means that KNN requires labeled data (i.e., data with known outcomes) for training, while K-means clustering does not.

Classification vs clustering: KNN is used for classification tasks, where the goal is to predict the class label of a new input based on its similarity to the labeled examples in the training set. K-means clustering, on the other hand, is used for clustering tasks, where the goal is to group similar instances into clusters based on their feature similarity.

Distance metric: KNN uses a distance metric (such as Euclidean distance) to measure the similarity between two instances, while K-means clustering uses a clustering criterion (such as the sum of squared distances between points and their cluster centroid) to group instances into clusters.

Model complexity: KNN is a simple model that stores all of the training data and performs a similarity search at prediction time, while K-means clustering involves iterative updates to estimate the cluster centroids and assignments.

19. **How would you make a prediction using a Logistic Regression model?  Related To: Logistic Regression**

To make a prediction using a logistic regression model, you need to follow these steps:

Collect the input data: You need to collect the input data that you want to make predictions on. The input data should be in the same format and have the same features as the data used to train the logistic regression model.

Preprocess the input data: You may need to preprocess the input data to ensure that it is in the correct format and has the same scale as the training data. For example, you may need to normalize or standardize the data.

Compute the probability of the output: Using the logistic regression model, compute the probability of the output variable taking the value of 1, given the input data. This is done using the logistic function, which maps the input data to a probability between 0 and 1\.

Make a binary prediction: To make a binary prediction (i.e., to classify the input data as belonging to one of two classes), you need to set a threshold value for the probability. If the probability is above the threshold, you predict that the output variable takes the value of 1, otherwise you predict that it takes the value of 0\.

The specific threshold value used may depend on the problem and the trade-off between false positives and false negatives. In some cases, it may be better to use a threshold that maximizes a specific performance metric, such as accuracy or F1 score, on a validation set.

Overall, making a prediction using a logistic regression model involves computing the probability of the output variable taking the value of 1, given the input data, and making a binary prediction based on a threshold value.

20. **How do you choose the optimal k in k-NN?  Related To: K-Nearest Neighbors**

Choosing the optimal value of k in k-NN depends on the dataset and the problem at hand. Generally, a higher value of k leads to a smoother decision boundary, but may lead to increased bias. A lower value of k leads to a more complex decision boundary, but may lead to increased variance and overfitting.

One common approach to choosing the optimal value of k is to use cross-validation. In k-fold cross-validation, the dataset is divided into k equal-sized folds. The algorithm is trained on k-1 folds and tested on the remaining fold. This process is repeated k times, with each fold being used for testing once. The performance of the algorithm is then averaged over the k folds.

To choose the optimal value of k, the algorithm is trained and tested using different values of k, and the performance is evaluated using cross-validation. The value of k that gives the best performance on the validation set is then chosen.

Another approach to choosing the optimal value of k is to use a grid search. In a grid search, the algorithm is trained and tested using different values of k, and the performance is evaluated using a validation set. The value of k that gives the best performance on the validation set is then chosen.

It is also important to note that the choice of k may depend on the size and distribution of the dataset, as well as the number of features and the noise in the data. Therefore, it is often recommended to try different values of k and compare their performance on a validation set.

21. **Why would you use the Kernel Trick?  Related To: SVM**

The kernel trick is a technique used in support vector machines (SVM) to transform the data into a higher-dimensional space in order to find a non-linear decision boundary. The kernel function computes the dot product between two points in the higher-dimensional space without actually computing the coordinates of the points in that space.

The kernel trick is useful in cases where the data is not linearly separable in the original feature space. By transforming the data into a higher-dimensional space, the kernel trick allows the SVM to find a decision boundary that can separate the classes.

The kernel trick is computationally efficient because it avoids the need to explicitly compute the coordinates of the points in the higher-dimensional space. Instead, it only needs to compute the dot products between pairs of points in the original feature space, which is often much faster.

In summary, the kernel trick is used in SVMs to handle non-linearly separable data by transforming it into a higher-dimensional space where a linear decision boundary can be found. The kernel function is used to compute the dot product between pairs of points in the higher-dimensional space efficiently without actually computing their coordinates.

22. **What types of Classification Algorithms do you know?**  

There are various classification algorithms that can be used for supervised learning tasks, including:

Logistic Regression  
Naive Bayes  
Decision Trees  
Random Forest  
Support Vector Machines (SVM)  
K-Nearest Neighbors (KNN)  
Gradient Boosting

23. **What's the difference between Multiclass Classification models and Multi Label model?  Related To: ML Design Patterns**

Multiclass classification models and multi-label classification models are both used for supervised learning tasks where the goal is to predict the target variable based on input features. However, there are some differences between them.

Multiclass classification models are used when there are three or more possible target classes, and the goal is to predict the single most likely class for each data point. For example, a multiclass classification model can be used to predict the type of animal in an image, where the possible classes are cat, dog, and bird. The model would output a single predicted class label for each image.

On the other hand, multi-label classification models are used when there are multiple possible target labels for each data point. In this case, the model outputs a set of binary labels that indicate the presence or absence of each possible label for a given data point. For example, a multi-label classification model can be used to predict the topics of a news article, where the possible labels are sports, politics, and entertainment. The model would output a set of binary labels indicating which topics the article is about.

24. **What is the Hinge Loss in SVM?  Related To: SVM, Cost Function**

In SVM, the Hinge Loss is a commonly used cost function that is used to train the model and find the optimal hyperplane that separates the data points into different classes.

25. **What is the difference between a Weak Learner vs a Strong Learner and why they could be useful?  Related To: Ensemble Learning**

In ensemble learning, a weak learner is a model that performs only slightly better than random guessing, while a strong learner is a model that performs significantly better than random guessing.

Weak learners are useful because they can be combined in various ways to form a strong learner. For example, in boosting, a sequence of weak learners is trained, with each learner trying to correct the errors made by the previous learner. The final model is a weighted combination of these weak learners, where the weights are determined by their individual performance. This process of boosting can convert a set of weak learners into a strong learner that can generalize well to new data.

On the other hand, strong learners are useful because they can achieve high accuracy on the training data and also generalize well to new data. However, they may be more prone to overfitting and may require more computational resources to train. In some cases, it may be difficult to find a single strong learner that performs well on all types of data, which is where ensemble methods come in.

26. **What's the difference between Bagging and Boosting algorithms?  Related To: Ensemble Learning, Data Processing, Bias & Variance**

Bagging and boosting are two popular ensemble learning techniques used to improve the accuracy and robustness of machine learning models.

Bagging (Bootstrap Aggregating) is a technique where multiple independent models are trained on different random samples (with replacement) of the training data. The predictions of these models are then combined (averaged or majority voting) to make a final prediction. Bagging reduces the variance of the model, making it less likely to overfit to the training data. Bagging is often used with decision trees or random forests.

Boosting, on the other hand, is a technique where a sequence of models are trained iteratively, with each model trying to correct the errors made by the previous model. The weights of the training samples are adjusted at each iteration to focus more on the samples that were misclassified in the previous iteration. Boosting increases the accuracy of the model, but can also increase its variance, making it more prone to overfitting. Popular boosting algorithms include AdaBoost, Gradient Boosting, and XGBoost.

The main difference between bagging and boosting lies in the way they combine the predictions of multiple models. Bagging combines the predictions of multiple independent models in parallel, while boosting combines the predictions of multiple models sequentially, with each model learning from the errors of the previous model.

In terms of bias and variance, bagging tends to reduce variance and increase bias, while boosting tends to reduce bias and increase variance. Bagging reduces variance by averaging out the predictions of multiple models, but this can also introduce some bias. Boosting reduces bias by focusing more on the misclassified samples, but this can also introduce some variance.

Overall, both bagging and boosting can be useful techniques for improving the performance of machine learning models, and the choice between them depends on the specific problem and the characteristics of the data.

27. **Can you choose a classifier based on the size of the training set?  Related To: Naïve Bayes**

Yes, the choice of classifier can depend on the size of the training set. Naïve Bayes is known to work well even with small training sets because it makes a strong assumption of independence between features, which can help reduce the risk of overfitting. However, as the size of the training set increases, other classifiers such as logistic regression or support vector machines may perform better due to their ability to capture more complex relationships between features. Therefore, the choice of classifier should be based on a combination of factors including the size of the training set, the complexity of the problem, and the desired accuracy and interpretability of the model.

28. **How would you use Naive Bayes classifier for categorical features? What if some features are numerical?  Related To: Naïve Bayes**

Naive Bayes classifier is a probabilistic algorithm that can be used for both categorical and numerical features.

For categorical features, the Naive Bayes algorithm assumes that each feature is independent of every other feature, given the class label. The algorithm calculates the probability of each class label given the feature values, and then selects the label with the highest probability as the predicted class.

For numerical features, the algorithm assumes that the feature follows a specific probability distribution, such as Gaussian or Bernoulli, and calculates the conditional probability of each feature given the class label. The algorithm then multiplies the conditional probabilities of each feature to obtain the joint probability of the class label and the feature values, and selects the label with the highest probability as the predicted class.

If the data contains a mixture of categorical and numerical features, the Naive Bayes algorithm can still be used by converting the numerical features into categorical features. This can be done by dividing the numerical values into discrete bins and assigning each value to the corresponding bin. Alternatively, the numerical features can be converted into categorical features by using a threshold or some other criteria to divide them into binary values, such as high/low, positive/negative, etc. Once all features are categorical, the Naive Bayes algorithm can be applied as usual.

29. **How does the AdaBoost algorithm work?  Related To: Ensemble Learning**

AdaBoost (Adaptive Boosting) is an ensemble learning algorithm that combines multiple weak learners to create a strong learner. The algorithm works as follows:

Initialize the weights of each sample in the training set to 1/n, where n is the total number of samples.  
Train a weak learner (e.g., decision tree) on the training set and calculate its error rate.  
Increase the weights of the misclassified samples and decrease the weights of the correctly classified samples.  
Repeat steps 2 and 3 for a predetermined number of iterations (or until the desired level of accuracy is achieved).  
Combine the weak learners using a weighted average, where the weights are based on the performance of each learner.  
During each iteration, the algorithm places more emphasis on the misclassified samples, which improves the performance of the subsequent weak learners. This adaptive weighting scheme allows AdaBoost to focus on the most difficult samples in the training set, leading to a strong classifier with high accuracy.

30. **How does ROC curve and AUC value help measure how good a model is?  Related To: Model Evaluation**

ROC (Receiver Operating Characteristic) curve and AUC (Area Under the Curve) value are commonly used to evaluate the performance of binary classification models.

The ROC curve is a plot of the True Positive Rate (TPR) against the False Positive Rate (FPR) at different classification thresholds. The TPR is the ratio of correctly predicted positive samples to the total positive samples, while the FPR is the ratio of incorrectly predicted positive samples to the total negative samples. The ROC curve provides a visual representation of how well the model can distinguish between the positive and negative classes. A model with a better performance will have a curve that is closer to the top left corner of the plot.

The AUC value, on the other hand, provides a single scalar value to represent the performance of the model. It measures the area under the ROC curve, which ranges from 0 to 1\. A perfect classifier will have an AUC of 1, while a random classifier will have an AUC of 0.5. A higher AUC value indicates a better performance of the model in terms of its ability to distinguish between the two classes.

31. **What's the difference between Softmax and Sigmoid functions?  Related To: Logistic Regression**

32. **Name some classification metrics and when would you use each one**


33. **Provide an intuitive explanation of Linear Support Vector Machines (SVMs)  Related To: SVM**

34. **Could you convert Regression into Classification and vice versa?**


35. **What's the difference between One-vs-Rest and One-vs-One?  Related To: Data Processing**

36. **How does the Naive Bayes classifier work?  Related To: Naïve Bayes**

37. **How do you use a supervised Logistic Regression for Classification?  Related To: Logistic Regression, Supervised Learning**

38. **What is a Confusion Matrix?  Related To: Supervised Learning, Model Evaluation**

39. **What's the difference between Generative Classifiers and Discriminative Classifiers? Name some examples of each one  Related To: Naïve Bayes**

40. **What are some advantages and disadvantages of using AUC to measure the performance of the model?  Related To: Model Evaluation**

41. **What is the F-Score?  Related To: Statistics, Model Evaluation**

42. **Name some advantages of using Support Vector Machines vs Logistic Regression for classification  Related To: SVM, Logistic Regression**

43. **How is AUC \- ROC curve used in classification problems?   Related To: Model Evaluation**

44. **When would you use SVM vs Logistic regression?  Related To: SVM, Logistic Regression**

45. **How would you use a Confusion Matrix for determining a model performance?  Related To: Data Processing, Model Evaluation**

46. **Compare Naive Bayes vs with Logistic Regression to solve classification problems  Related To: Logistic Regression, Naïve Bayes**

47. **What are the trade-offs between the different types of Classification Algorithms? How would do you choose the best one?  Related To: Ensemble Learning, Naïve Bayes**

48. **How would you Calibrate Probabilities for a classification model?  Related To: Probability**

49. **Are there any problems using Naive Bayes for Classification?  Related To: Naïve Bayes**

50. **What's the difference between Random Oversampling and Random Undersampling and when they can be used?  Related To: Data Processing**

51. **How would you deal with classification on Non-linearly Separable data?  Related To: SVM**

52. **How would you choose an evaluation metric for an Imbalanced classification?  Related To: Model Evaluation**

53. **What is AIC?  Related To: Model Evaluation**

54. **Can Logistic Regression be used for an Imbalanced Classification problem?  Related To: Logistic Regression**

55. **What's the difference between ROC and Precision-Recall Curves?  Related To: Model Evaluation**

56. **How to interpret F-measure values?  Related To: Model Evaluation**

57. **Why would you use Probability Calibration?  Related To: Probability**

58. **Define what is Clustering?**


59. **Give examples of using Clustering to solve real-life problems**


60. **What are Self-Organizing Maps?  Related To: Neural Networks**

61. **Why do you need to perform Significance Testing in Clustering?**


62. **What is the Jaccard Index?**


63. **What is Similarity-based Clustering?**


64. **What is Mean-Shift Clustering?**


65. **What is the difference between a Multiclass problem and a Multilabel problem?  Related To: Supervised Learning**

66. **What is the difference between the two types of Hierarchical Clustering?  Related To: Unsupervised Learning**

67. **What is the Mixture in Gaussian Mixture Model?**


68. **How can Evolutionary Algorithms be used for Clustering?  Related To: Genetic Algorithms**

69. **What would be a good way to use Clustering for Outlier detection?  Related To: Anomaly Detection**

70. **What are some of the differences between Anomaly Detection and Behaviour Detection?  Related To: Anomaly Detection**

71. **What is the difference between Cost Function vs Gradient Descent?  Related To: Gradient Descent**

72. **Explain the steps of k-Means Clustering Algorithm  Related To: K-Means Clustering**

73. **Provide an analogy for a Cost Function in real life**


74. **Explain what is Cost (Loss) Function in Machine Learning?**


75. **What is the difference between Objective function, Cost function and Loss function**


76. **What is the Hinge Loss in SVM?  Related To: SVM, Classification**

77. **Why don’t we use Mean Squared Error as a cost function in Logistic Regression?  Related To: Logistic Regression**

78. **What type of Cost Functions do Greedy Splitting use?  Related To: Decision Trees**

79. **How would you fix Logistic Regression Overfitting problem?  Related To: Linear Regression, Model Evaluation**

80. **How would you choose the Loss Function for a Deep Learning model?  Related To: Deep Learning**

81. **What is the Objective Function of k-Means?  Related To: K-Means Clustering**

82. **What Distance Function do you use for Quantitative Data?  Related To: Data Processing**

83. **What are some necessary Mathematical Properties a Loss Function needs to have in Gradient-Based Optimization?  Related To: Gradient Descent**

84. **What are some different types of Clustering Structures that are used in Clustering Algorithms?**


85. **What is Latent Class Model?**


86. **What is Silhouette Analysis?**


87. **What is the difference between the Manhattan Distance and Euclidean Distance in Clustering?  Related To: K-Means Clustering**

88. **When would you use Hierarchical Clustering over Spectral Clustering?**


89. **Name some pros and cons of Mean Shift Clustering** 

    

90. **While performing K-Means Clustering, how do you determine the value of K?  Related To: K-Means Clustering**

91. **Compare Hierarchical Clustering and k-Means Clustering  Related To: K-Means Clustering**

92. **Where do the Similarities come from in Similarity-based Clustering?**   
      
       
93. **What is a Mixture Model?  Related To: K-Means Clustering**

94. **How would you perform an Observation-Based Clustering for Time-Series Data?**


95. **What is the difference and connection between Clustering and Dimension Reduction?  Related To: Unsupervised Learning, Dimensionality Reduction**

96. **Why does K-Means have a higher bias when compared to Gaussian Mixture Model?  Related To: Unsupervised Learning**

97. **Explain how a cluster is formed in the DBSCAN Clustering Algorithm  Related To: Unsupervised Learning**

98. **Why is Euclidean Distance not good for Sparse Data?**


99. **How to tell if data is clustered enough for clustering algorithms to produce meaningful results?  Related To: K-Means Clustering, Unsupervised Learning**

100. **What makes the distance measurement of k-Medoids better than k-Means?  Related To: K-Means Clustering**  
101. **When would you use Hierarchical Clustering over k-Means Clustering?  Related To: K-Means Clustering**

102. **How does Cluster Algorithms work on detecting Anomalies when the cluster sizes are different?  Related To: Anomaly Detection**

103. **When using various Clustering Algorithms, why is Euclidean Distance not a good metric in High Dimensions?** 

      

104. **Explain the Dirichlet Process Gaussian Mixture Model**


105. **How to choose among the various clustering Distance Measures?**


106. **What are some characteristics of Clustering Algorithms concerning Anomaly Detection?  Related To: Anomaly Detection**

107. **How would you choose the number of Clusters when designing a K-Medoid Clustering Algorithm?  Related To: K-Means Clustering**

108. **When would you use Segmentation over Clustering?**   
       
        
109. **What is the motivation behind the Expectation-Maximization Algorithm?**


110. **Explain the different frameworks used for k-Means Clustering  Related To: K-Means Clustering**

111. **What is the relationship between k-Means Clustering and PCA?  Related To: K-Means Clustering, PCA**

112. **What are Decision Trees?  Related To: Supervised Learning**

113. **Explain the structure of a Decision Tree  Related To: Supervised Learning**

114. **What are some advantages of using Decision Trees?**


115. **How is a Random Forest related to Decision Trees?  Related To: Ensemble Learning, Random Forest**

116. **How are the different nodes of decision trees represented?  Related To: Supervised Learning**

117. **What type of node is considered Pure?**


118. **How would you deal with an Overfitted Decision Tree?**


119. **What are some disadvantages of using Decision Trees and how would you solve them?  Related To: Ensemble Learning, Data Processing**

120. **What type of Cost Functions do Greedy Splitting use?  Related To: Cost Function**

121. **What is Gini Index and how is it used in Decision Trees?**


122. **What is the Chi-squared test?**


123. **How does the CART algorithm produce Regression Trees?**


124. **What is Tree Bagging?  Related To: Ensemble Learning**

125. **What is the difference between OOB score and validation score?  Related To: Ensemble Learning, Random Forest**

126. **What is Greedy Splitting?**


127. **Why do you need to Prune the decision tree?**


128. **How does the CART algorithm produce Classification Trees?**


129. **How would you define the Stopping Criteria for decision trees?**   
        
130. **What is Entropy?  Related To: Machine Learning**

131. **How do we measure the Information?  Related To: Machine Learning**

132. **What is the difference between Post-pruning and Pre-pruning?**


133. **Compare Linear Regression and Decision Trees  Related To: Linear Regression**

134. **What is Tree Boosting?  Related To: Ensemble Learning**

135. **How would you tune a Random Forest algorithm to improve its performance?**


136. **Imagine that you know there are outliers in your data, would you use Logistic Regression?  Related To: Anomaly Detection, Logistic Regression, Random Forest**

137. **What are some disadvantages of the CHAID algorithm?**


138. **Explain how ID3 produces classification trees?**


139. **Compare ID3 and C4.5 algorithms**


140. **What is the relationship between Information Gain and Information Gain Ratio?**


141. **Compare Decision Trees and k-Nearest Neighbors  Related To: K-Nearest Neighbors**

142. **While building Decision Tree how do you choose which attribute to split at each node?**  
       
         
143. **How would you compare different Algorithms to build Decision Trees?**    
144. **How do you Gradient Boost decision trees?  Related To: Ensemble Learning**

145. **What are the differences between Decision Trees and Neural Networks?  Related To: Neural Networks**

146. **How to use Isolation Forest for Anomalies detection?  Related To: Anomaly Detection**

147. **What is the difference between Gini Impurity and Entropy in Decision Tree?**


148. **When should I use Gini Impurity as opposed to Information Gain (Entropy)?**


149. **Explain the CHAID algorithm**


150. **Explain how can CART algorithm perform Pruning?**


151. **Compare C4.5 and C5.0 algorithms**


152. **What is the Variance Reduction metric in Decision Trees?**


153. **What is the use of Entropy pertaining to Decision Trees?  Related To: Machine Learning**

154. **Compare Decision Trees and Logistic Regression  Related To: Logistic Regression**

155. **What is the difference between Gradient Boosting and Adaptive Boosting?  Related To: Supervised Learning**

156. **What are the steps for Binary Recursive Partitioning in Decision Trees?**


157. **Explain the measure of goodness used by CART**


158. **How do you extend Decision Trees to Collaborative Filtering?  Related To: Recommendation Systems**

159. **Decision Trees Practical Challenges**

160. **How to extract the decision rules from Scikit-learn decision tree?  Related To: Scikit-Learn**

161. **Explain how you would implement CART training algorithm in plain Python PY Related To: Python**

162.   **What is the Curse of Dimensionality and how can Unsupervised Learning help with it?  Related To: Unsupervised Learning, Curse of Dimensionality**

163. **What is Principal Component Analysis (PCA)?  Related To: PCA**

164. **Explain how do you understand Dimensionality Reduction  Related To: Data Processing**

165. **Explain One-Hot Encoding and Label Encoding. Does the dimensionality of the dataset increase or decrease after applying these techniques?  Related To: Feature Engineering**  
166. **Why is data more sparse in a high-dimensional space?  Related To: Data Processing, Curse of Dimensionality**

167. **How is the first principal component axis selected in PCA?  Related To: PCA**

168. **How does the Curse of Dimensionality affect Machine Learning models?  Related To: Curse of Dimensionality**

169. **What methods of Hyperparameters Tuning do you know?  Related To: Optimisation, ML Design Patterns, Model Evaluation**

170. **When would you use Grid Search vs Random Search for Hyperparameter Tuning?  Related To: Optimisation, ML Design Patterns**

171. **How does an Isomap perform Dimensionality Reduction?**


172. **What are the two branches of Dimensionality Reduction?**


173. **Why is Centering and Scaling the data important before performing PCA?  Related To: Linear Algebra, PCA**

174. **What is Singular Value Decomposition?**


175. **What is the difference between PCA and Random Projection approaches?  Related To: PCA**

176. **What are some advantages of using LLE over PCA?  Related To: Unsupervised Learning, PCA**

177. **How does Random Projection reduce the dimensionality of a set of points?**


178. **Explain the Sparse Random Projection**


179. **How does High Dimensionality affect Distance-Based Mining Applications?  Related To: Data Mining, Curse of Dimensionality**

180. **How does the Curse of Dimensionality affect Privacy Preservation?  Related To: Curse of Dimensionality**

181. **What is the Crowding Problem?**


182. **Does kNN suffer from the Curse of Dimensionality and if it why?  Related To: K-Nearest Neighbors, Curse of Dimensionality**

183. **How many Dimensionality Reduction Techniques do you know?**


184. **What is the difference and connection between Clustering and Dimension Reduction?  Related To: Clustering, Unsupervised Learning**

185. **What's the difference between PCA and t-SNE?  Related To: PCA**  
186. **Explain the Locally Linear Embedding algorithm for Dimensionality Reduction  Related To: Unsupervised Learning**

187. **What is Multidimensional Scaling?  Related To: Data Processing**

188. **What is Sparse PCA?  Related To: PCA**

189. **When would you use Manifold Learning techniques over PCA?  Related To: PCA**

190. **What is Kernel Principal Component Analysis?**


191. **What is t-Distributed Stochastic Neighbour Embedding?**


192. **What are the rules for generating a random matrix when Gaussian Random Projection is used?**


193. **How does the Curse of Dimensionality affect k-Means Clustering?  Related To: K-Means Clustering, Curse of Dimensionality**

194. **How does a Deep Neural Network escape/resist the Curse of Dimensionality?  Related To: Deep Learning, Curse of Dimensionality**

195. **Why does the hyperparameter optimisation method GridSearch suffer from the Curse of Dimensionality?  Related To: Curse of Dimensionality**

196. **Does Random Forest suffer from the Curse of Dimensionality?  Related To: Curse of Dimensionality, Random Forest**

197. **Does linear SVMs suffer from the Curse of Dimensionality?  Related To: SVM, Curse of Dimensionality**

198. **How does Isometric Mapping (Isomap) work?**


199. **What is Independent Component Analysis?**   
       
        
200. **What is the Hughes Phenomenon?  Related To: Data Structures**

201. **How does Normalization reduce the Dimensionality of the Data if you project the data to a Unit Sphere?  Related To: Data Processing**

202. **How does the Metric Multidimensional Scaling (MDS) algorithm reduce dimensionality?**    
203. **What is Ensemble Learning?**   
        
204. **How would you define Random Forest?  Related To: Random Forest**

205. **What are Weak Learners?  Related To: Machine Learning**

206. **What are Ensemble Methods?  Related To: Random Forest**  
207. **How is a Random Forest related to Decision Trees?  Related To: Decision Trees, Random Forest**

208. **What is Meta-Learning?**


209. **What are the differences between Bagging and Boosting?**  
       
       
210. **How does Stacking work?**


211. **Since Ensemble Learning provides better output most of the time, why do you not use it all the time?**


212. **Would it defeat the purpose of Ensemble Learning to exclude Outliers?**


213. **What are the differences between Homogeneous and Heterogeneous Ensembles?**


214. **Is Random Forest an Ensemble Algorithm?  Related To: Random Forest**

215. **What are some Real-World Applications of Ensemble Learning?**


216. **How does Ensemble Systems help in Incremental Learning?  Related To: Deep Learning**

217. **What is a Super Learner Algorithm?**


218. **What is the difference between a Weak Learner vs a Strong Learner and why they could be usefu?  Related To: Classification**

219. **What's the difference between Bagging and Boosting algorithms?  Related To: Data Processing, Classification, Bias & Variance**

220. **How does the AdaBoost algorithm work?  Related To: Classification**

221. **What are some disadvantages of using Decision Trees and how would you solve them?  Related To: Decision Trees, Data Processing**

222. **What is Tree Bagging?  Related To: Decision Trees**

223. **What's the similarities and differences between Bagging, Boosting, Stacking?  Related To: Machine Learning**

224. **Explain the concept behind BAGGing  Related To: Random Forest**

225. **What is the difference between OOB score and validation score?  Related To: Decision Trees, Random Forest**

226. **What is Tree Boosting?  Related To: Decision Trees**

227. **How is Gradient Boosting used to improve Supervised Learning?  Related To: Supervised Learning**  
228. **What is the difference between Ensemble Learning and Multiple Kernel Learning?**


229. **Explain the architecture of a Super Learner Algorithm**


230. **What is the Bagging Algorithm?**


231. **What are some variants of the AdaBoost Algorithm?**


232. **Why do Ensemble Models work better when the Models have Low Correlation?**


233. **In what situations do you not use Ensemble Classifiers?**


234. **What are Ensemble Nystrom Algorithms?**


235. **What is the process of building an Ensemble System?**


236. **Is it posible to apply Ensemble Learning methods to the Quantile Regression Problem?**  
       
       
237. **Why is Model Stacking effective in improving Performance?**


238. **What are the trade-offs between the different types of Classification Algorithms? How would do you choose the best one?  Related To: Naïve Bayes, Classification**

239. **How do you Gradient Boost decision trees?  Related To: Decision Trees**

240. **Give some reasons to choose Random Forests over Neural Networks  Related To: Neural Networks, Random Forest**

241. **Can you use the LASSO method for Base Learner Selection?**


242. **How are Ensemble Methods used with Deep Neural Networks?  Related To: Deep Learning**

243. **What are the methods to evaluate Ensembles of Classifiers?**


244. **Explain Discrete AdaBoost Algorithm**


245. **How does Ensemble Learning tackle the No-Free Lunch Dilemma?**


246. **How do you decide to use between Gradient Boosting Trees and Random Forest?**


247. **How would you find the optimal number of random features to consider at each split?  Related To: Random Forest**

248. **Summarize the Statistical, Computational, and Representational motivation of Ensemble Learning**


249. **How would you apply the Standard Mixture Models in the context of Regression?**


250. **Explain mathematically the Ensemble Nystrom algorithm**


251. **How is Computational Complexity measured in Ensemble Learning?  Related To: Data Mining**

252. **What are some common Machine Learning problems that Unsupervised Learning can help with?  Related To: Unsupervised Learning**

253. **What's the difference between Feature Engineering vs. Feature Selection?  Related To: Data Processing**

254. **What advantages does Deep Learning have over Machine Learning?  Related To: Deep Learning**

255. **Name some benefits of Feature Selection   Related To: Data Processing**

256. **Explain One-Hot Encoding and Label Encoding. Does the dimensionality of the dataset increase or decrease after applying these techniques?  Related To: Dimensionality Reduction**

257. **Can we use PCA for feature selection?  Related To: PCA**

258. **What's the difference between Forward Feature Selection and Backward Feature Selection?**


259. **What are the basic objects that a ML pipeline should contain?  Related To: Scikit-Learn**

260. **How do you use the F-test to select features?** 


261. **What are some recommended choices for Imputation Values?  Related To: ML Design Patterns, Data Processing**

262. **How would you improve the performance of Random Forest?  Related To: Random Forest**

263. **How would you decide on the importance of variables for the Multivariate Regression model?   Related To: Linear Regression**

264. **How has Translation of words improved from the Traditional methods?  Related To: NLP, Neural Networks**

265. **How to choose the features for a Neural Network?  Related To: Data Processing, Neural Networks**

266. **When would you remove Correlated Variables?  Related To: Data Processing**

267. **How do you perform End of Tail Imputation?**


268. **What Feature Selection methods do you know?**   
       
        
269. **How do you perform feature selection with Categorical Data?**


270. **How do you transform a Skewed distribution into a Normal distribution?  Related To: Statistics**

271. **How do you perform Principal Component Analysis (PCA)?  Related To: PCA**

272. **Explain the Stepwise Regression technique  Related To: Linear Regression**

273. **How does the Recursive Feature Elimination (RFE) work?**


274. **Why would you use Permutation Feature Importance and how does this algorithm work?**


275. **What's the difference between Principal Component Analysis and Independent Component Analysis?  Related To: PCA**

276. **How to perform Feature Engineering on Unknown features?  Related To: Data Processing**

277. **When would you use Fine-Tuning vs Feature Extraction in Transfer Learning?  Related To: ML Design Patterns, Data Processing**

278. **What is the idea behind the Gradient Descent?**


279. **Explain the intuition behind Gradient Descent algorithm  Related To: Linear Regression**

280. **What is the difference between Cost Function vs Gradient Descent?  Related To: Cost Function**

281. **What is the difference between Maximum Likelihood Estimation and Gradient Descent?**   
       
        
282. **Can Gradient Descent be applied to Non-Convex Functions?**


283. **What are some types of Gradient Descent do you know?**


284. **Compare the Mini-batch Gradient Descent, Stochastic Gradient Descent, and Batch Gradient Descent**


285. **Explain how does the Gradient descent work in Linear Regression  Related To: Linear Regression**

286. **In which case you would use Gradient Descent method or Ordinary Least Squares and why?  Related To: Linear Regression**

287. **Name some Evaluation Metrics for Regression Model and when you would use one?  Related To: Linear Regression**

288. **How does the Adam method of Stochastic Gradient Descent work?  Related To: Optimisation**  
289. **How does Batch Size affect the Convergence of SGD and why?  Related To: Optimisation**

290. **When should we use Algorithms like Adam as opposed to SGD?  Related To: Optimisation**

291. **Compare Batch Gradient Descent and Stochastic Gradient Descent  Related To: Optimisation**

292. **When Optimizing a Neural Network, how do you define the Termination Condition for Gradient Descent?  Related To: Optimisation**

293. **How do Gradient-Based Algorithms deal with the flat regions with desired points?**


294. **What are some necessary Mathematical Properties a Loss Function needs to have in Gradient-Based Optimization?  Related To: Cost Function**

295. **What is the difference between Gradient Descent and Stochastic Gradient Descent?**


296. **Does Gradient Descent always converge to an optimum?**


297. **How is the Adam Optimization Algorithm different when compared to Stochastic Gradient Descent?**


298. **Name some advantages of using Gradient descent vs Ordinary Least Squares for Linear Regression  Related To: Linear Regression**

299. **What is the difference between Momentum based Gradient Descent and Nesterov's Accelerated Gradient Descent?  Related To: Optimisation**

300. **In what situations would you prefer Coordinate Descent over Gradient Descent?  Related To: Optimisation**

301. **What is the Mirror Descent?  Related To: Optimisation**

302. **Explain in detail how Momentum-based Gradient Descent and Nesterov's Accelerated Gradient Descent work  Related To: Optimisation**

303. **Explain Mathematically, how Stochastic Gradient Descent saves time compared to Standard Gradient Descent.**  
         
304. **For both Convex, and Non-Convex Problems, does the Gradient in SGD always point to the global extreme value?  Related To: Optimisation**

305. **What are some applications of k-Means Clustering?**


306. **What is the difference between KNN and K-means Clustering?  Related To: Supervised Learning, Unsupervised Learning, Classification**

307. **Explain what is k-Means Clustering?**


308. **What is the Uniform Effect that k-Means Clustering tends to produce?**


309. **What are some Stopping Criteria for k-Means Clustering?**


310. **Explain the steps of k-Means Clustering Algorithm  Related To: Cost Function**

311. **Why does k-Means Clustering use mostly the Euclidean Distance metric?**   
       
312. **What is the main difference between k-Means and k-Nearest Neighbours?  Related To: K-Nearest Neighbors**

313. **How does K-Means perform Clustering?  Related To: Unsupervised Learning**

314. **What is the use of Fuzzy C-Means Clustering?**


315. **How would you Pre-Process the data for k-Means?**


316. **Explain some cases where k-Means clustering fails to give good results  Related To: Anomaly Detection**

317. **How would you perform k-Means on very large datasets?**


318. **What is the difference between traditional k-Means and the SAIL algorithm?**


319. **What is the difference between Classical k-Means and Spherical k-Means?**


320. **What is the k-Means based Consensus Clustering?**


321. **What is the difference between the Manhattan Distance and Euclidean Distance in Clustering?  Related To: Clustering**

322. **How to determine k using the Elbow Method? PY Related To: Python**

323. **What is the difference between k-Means and k-Medians and when would you use one over another?**


324. **How do you measure the Consistency between different k-Means Clustering outputs?**


325. **What is the Objective Function of k-Means?  Related To: Cost Function**

326. **Can you find Outliers using k-Means?  Related To: Anomaly Detection**

327. **While performing K-Means Clustering, how do you determine the value of K?  Related To: Clustering**

328. **Compare Hierarchical Clustering and k-Means Clustering  Related To: Clustering**

329. **What is a Mixture Model?  Related To: Clustering**

330. **How is Entropy used as a Clustering Validation Measure?**    
331. **How to tell if data is clustered enough for clustering algorithms to produce meaningful results?  Related To: Clustering, Unsupervised Learning**

332. **How does Spherical k-Means work with high dimensional data such as text?**


333. **What makes the distance measurement of k-Medoids better than k-Means?  Related To: Clustering**

334. **When would you use Hierarchical Clustering over k-Means Clustering?  Related To: Clustering**

335. **How to determine k using the Silhouette Method? PY Related To: Python**

336. **When would you use Fuzzy C-Means as opposed to k-Means?**


337. **How do the clusters generated by k-Means and Mini Batch k-Means compare?**


338. **How does the Curse of Dimensionality affect k-Means Clustering?  Related To: Dimensionality Reduction, Curse of Dimensionality**

339. **How would you choose the number of Clusters when designing a K-Medoid Clustering Algorithm?  Related To: Clustering**

340. **How does Forgy Initialization, Random Partition Initialization, and kmeans++ Initialization compare with each other?**


341. **Why would you use Correlation-based Distances in k-Means Clustering?**


342. **What are some properties of the Point-to-Centroid Distance?**


343. **Explain the different frameworks used for k-Means Clustering  Related To: Clustering**

344. **What is the relationship between k-Means Clustering and PCA?  Related To: Clustering, PCA**

345.  **How do you choose the optimal k in k-NN?  Related To: Classification**

346. **Would you use K-NN for large datasets?  Related To: Data Processing**

347. **What's the difference between k-Nearest Neighbors and Radius Nearest Neighbors?**


348. **What is k-Nearest Neighbors algorithm?  Related To: Supervised Learning**

349. **What is the main difference between k-Means and k-Nearest Neighbours?  Related To: K-Means Clustering**

350. **Compare K-Nearest Neighbors (KNN) and SVM  Related To: SVM**

351. **What are some advantages and disadvantages of k-Nearest Neighbors?**    
352. **How can you relate the KNN Algorithm to the Bias-Variance tradeoff?  Related To: Bias & Variance**

353. **How do you select the value of K for k-Nearest Neighbors?**


354. **If you are using k-Nearest Neighbors, what type of Normalization should be used?  Related To: Data Processing**

355. **Does kNN suffer from the Curse of Dimensionality and if it why?  Related To: Dimensionality Reduction, Curse of Dimensionality**

356. **Compare Decision Trees and k-Nearest Neighbors  Related To: Decision Trees**

357. **How do you know if a system of two linear equations has one solution, multiple solutions or no solutions?**


358. **What's the difference between Cross Product and Dot Product?**


359. **What is Frobenius norm?**


360. **How do you find eigenvalues of a matrix? Could you provide an example?**


361. **What is Ax=b? When does Ax=b has a unique solution?**


362.  **At what conditions does the inverse of a diagonal matrix exist?** 

 

363.  **What's the normal vector to the surface S provide below?**


364. **Why is Centering and Scaling the data important before performing PCA?  Related To: PCA, Dimensionality Reduction**

365. **Is the Eigendecomposition guaranteed to be unique for a real matrix? If not, then how do we represent it?**   
        
366. **What is the determinant of a square matrix? How is it calculated?** 


367. **Can the number of nonzero elements in a vector be defined as norm? If no, why?**


368. **How do you diagonalize a matrix?**


369. **What’s the difference between a Matrix and a Tensor?**


370. **What happens if we transform a vector z using a positive definite matrix?**


371. **What is Hadamard product of two matrices?**


372. **What is an Orthogonal Matrix? Why is computationally preferred?**   
        
373. **What are positive definite, negative definite, positive semi definite and negative semi definite matrices?**    
374. **At what conditions does the inverse of a matrix exist?**


375. **What are the conditions that a norm function has to satisfy?**


376. **How many ways of measure a vector do you know?**


377. **Diagonalize, if it's possible, the following matrix**


378. **Discuss span and linear dependence**


379. **What is broadcasting in connection to Linear Algebra?  Related To: NumPy**

380. **How to assign values to a MATLAB matrix on the diagonal with vectorization?  Related To: MATLAB**

381. **Given a matrix M, how do you calculate its Singular Value Decomposition?**


382. **What are Singular Eigenvalues, Left Singulars and Right Singulars?**


383. **When performing regularization, when would you choose L1-norm over L2-norm?**


384. **Why do we use Singular Value Decomposition? Why not just use Eigendecomposition?**


385. **Why would you use the Moore-Penrose Pseudoinverse and how would you calculate it?**


386. **What is Linear Regression?  Related To: Supervised Learning**

387. **How would you detect Overfitting in Linear Models?  Related To: Model Evaluation**

388. **What are types of Linear Regression?**


389. **How can you check if the Regression model fits the data well?** 


390. **Define Linear Regression and its structure  Related To: Supervised Learning**

391. **How does a Non-Linear regression analysis differ from Linear regression analysis?**


392. **What is the difference between Mean Absolute Error (MAE) vs Mean Squared Error (MSE)?**


393. **What's the difference between Covariance and Correlation?  Related To: Data Processing**

394. **Explain the intuition behind Gradient Descent algorithm  Related To: Gradient Descent**

395. **Provide an intuitive explanation of the Learning Rate?**


396. **How is the Error calculated in a Linear Regression model?**


397. **Explain what the Intercept Term means**


398. **Why use Root Mean Squared Error (RMSE) instead of Mean Absolute Error (MAE)?**


399. **Explain how does the Gradient descent work in Linear Regression  Related To: Gradient Descent**

400. **What are the Assumptions of Linear Regression?**    
401. **What is the difference between Ordinary Least Squares and Ridge Regression?**    
402. **How would you decide on the importance of variables for the Multivariate Regression model?   Related To: Feature Engineering**  
403. **What are the assumptions before applying the OLS estimator?  Related To: Data Processing**  
404. **What is the difference between a Regression Model and an ANOVA Model?**    
405. **How does it work the Backward Selection Technique?**    
406. **In which case you would use Gradient Descent method or Ordinary Least Squares and why?  Related To: Gradient Descent**  
407. **What are some challenges faced when using a Supervised Regression Model?  Related To: Supervised Learning**  
408. **Name a disadvantage of R-squared and explain how would you address it?**    
409. **What is the difference between Ordinary Least Squares and Lasso regression?**    
410. **How is Hypothesis Testing using in Linear Regression?**    
411. **Name some Evaluation Metrics for Regression Model and when you would use one?  Related To: Gradient Descent**  
412. **What is the difference between Linear Regression and Logistic Regression?  Related To: Logistic Regression**  
413. **Why would you use Normalisation vs Standardisation for Linear Regression?**    
414. **Why can't a Linear Regression be used instead of Logistic Regression?  Related To: Logistic Regression**  
415. **How would you fix Logistic Regression Overfitting problem?  Related To: Model Evaluation, Cost Function**  
416. **Compare Linear Regression and Decision Trees  Related To: Decision Trees**  
417. **What's the difference between Homoskedasticity and Heteroskedasticity?  Related To: Statistics**  
418. **How do you cope with Missing data in Regression?  Related To: Data Processing**  
419. **Explain the Stepwise Regression technique  Related To: Feature Engineering**  
420. **How would you detect Collinearity and what is Multicollinearity?**    
421. **How would you address the problem of Heteroskedasticity caused for a Measurement error?  Related To: Data Processing**  
422. **How would you deal with Outliers in your dataset?  Related To: Anomaly Detection, Data Processing**  
423. **How would you detect Heteroskedasticity?  Related To: Data Processing**  
424. **Name some advantages of using Gradient descent vs Ordinary Least Squares for Linear Regression  Related To: Gradient Descent**  
425. **How would you deal with Overfitting in Linear Regression models?  Related To: Model Evaluation**  
426. **Explain what is an Unrepresentative Dataset and how would you diagnose it?  Related To: Data Processing**  
427. **How would you compare models using the Akaike Information Criterion?**    
428. **How would you check if a Linear Model follows all Regression assumptions?**    
429. **How would you implement Linear Regression Function in SQL?  Related To: SQL**  
430. **What types of Robust Regression Algorithms do you know?  Related To: Anomaly Detection**  
431. **Provide an intuitive explanation of RANSAC Regression algorithm**    
432. **When Logistic Regression can be used?**    
433. **Why is Logistic Regression called Regression and not Classification?**    
434. **What is a Decision Boundary?  Related To: Classification**  
435. **How would you make a prediction using a Logistic Regression model?  Related To: Classification**  
436. **Why don’t we use Mean Squared Error as a cost function in Logistic Regression?  Related To: Cost Function**  
437. **What's the difference between Softmax and Sigmoid functions?  Related To: Classification**  
438. **Why is Logistic Regression considered a Linear Model?**    
439. **Compare SVM and Logistic Regression in handling outliers  Related To: SVM, Anomaly Detection**  
440. **What is the difference between Linear Regression and Logistic Regression?  Related To: Linear Regression**  
441. **Provide a mathematical intuition for Logistic Regression?**    
442. **Why can't a Linear Regression be used instead of Logistic Regression?  Related To: Linear Regression**  
443. **What can you infer from each of the hand drawn decision boundary of Logistic Regression below?**    
444. **How a Logistic Regression model is trained?**    
445. **How do you use a supervised Logistic Regression for Classification?  Related To: Supervised Learning, Classification**  
446. **In Logistic Regression, why is the Binary Cross-Entropy loss function convex?**    
447. **Explain the Vectorized Implementation of Logistic Regression?**    
448. **Explain the Space Complexity Analysis of Logistic Regression**     
449. **How can we avoid Over-fitting in Logistic Regression models?**    
450. **Imagine that you know there are outliers in your data, would you use Logistic Regression?  Related To: Anomaly Detection, Decision Trees, Random Forest**  
451. **Name some advantages of using Support Vector Machines vs Logistic Regression for classification  Related To: SVM, Classification**  
452. **When would you use SVM vs Logistic regression?  Related To: SVM, Classification**  
453. **Compare Naive Bayes vs with Logistic Regression to solve classification problems  Related To: Naïve Bayes, Classification**  
454. **Compare Decision Trees and Logistic Regression  Related To: Decision Trees**  
455. **Can Logistic Regression be used for an Imbalanced Classification problem?  Related To: Classification**  
456. **How many ways do you know to implement MLOps?**    
457. **What is a Model Registry and what are its benefits?**    
458. **What's the difference between Static Deployment and Dynamic Deployment?**    
459. **What production Testing methods do you know?**    
460. **What's the difference between Batch Processing and Stream Processing?**    
461. **What is Training-Serving Skew?**    
462. **How does the Champion-Challenger technique work?**    
463. **What is a structure of a typical ML Artifact?**    
464. **What are the pros and cons of using Microservices?**    
465. **Why you should package ML models?**    
466. **What Feature Attribution methods do you know?**    
467. **What's the difference between Continuous Integration, Continuous Delivery and Continuous Deployment?**    
468. **What kind of test can you perform in the MLOps cycle?**    
469. **What approaches can you take for testing the ML model during the training process?**    
470. **Why would you need a Model Store for your MLOps projects?**    
471. **What are some good practices for monitoring training in production?**    
472. **What are the pros and the cons of using Rolling Deployment?**    
473. **What are the benefits of using Blue-Green deployments?**    
474. **What are the steps in a basic ML Pipeline?**    
475. **How many ways of ML models packaging do you know?**    
476. **What are the benefits of CI/CD for Machine Learning systems?**    
477. **What are some good practices when performing testing in the MLOps cycle?**    
478. **Why would you use feature flags?**    
479. **Why would you monitor Feature Attribution instead of Feature Distribution?**    
480. **What types of Model Drift problems can you faced and how can you overcome it?**    
481. **What are the differences between Canary and Blue-Green strategies deployments?**    
482. **Why would you need to use a Feature Store service?**    
483. **How does the ML Test Score work?**    
484. **What's the difference between A/B testing model deployment strategy and Multi-Arm Bandit?**    
485. **When would you use Statistical Methods vs Statistical Process Control for Data Drift detection?**  
486. **What is a Naïve Bayes Classifier?**    
487. **Why Naive Bayes is called Naive?  Related To: Supervised Learning, Classification**  
488. **What are the basic components of a Content-Based System?  Related To: Recommendation Systems**  
489. **What Bayes' Theorem (Bayes Rule) is all about?**    
490. **Find a probability of dangerous Fire when there is Smoke  Related To: Probability**  
491. **Can you choose a classifier based on the size of the training set?  Related To: Classification**  
492. **How would you use Naive Bayes classifier for categorical features? What if some features are numerical?  Related To: Classification**  
493. **What are some advantages of using Naive Bayes Algorithm?**    
494. **How does the Naive Bayes classifier work?  Related To: Classification**  
495. **What are some disadvantages of using Naive Bayes Algorithm?**    
496. **What's the difference between Generative Classifiers and Discriminative Classifiers? Name some examples of each one  Related To: Classification**  
497. **What's the difference between the likelihood and the posterior probability in Bayesian statistics?  Related To: Probability**  
498. **How would you test hypotheses using Bayes' Rule?  Related To: Probability**  
499. **How do you use Naive Bayes model for Collaborative Filtering?  Related To: Recommendation Systems**  
500. **What is Bayesian Network?**    
501. **Compare Naive Bayes vs with Logistic Regression to solve classification problems  Related To: Logistic Regression, Classification**  
502. **What are the trade-offs between the different types of Classification Algorithms? How would do you choose the best one?  Related To: Ensemble Learning, Classification**  
503. **Are there any problems using Naive Bayes for Classification?  Related To: Classification**  
504. **Does Noisy Data benefit Bayesian?  Related To: Data Mining**  
505. **Why to use NumPy?**    
506. **Compute the min/max for each row for a NumPy 2D array**    
507. **What is the difference between ndarray and array in NumPy?**    
508. **Explain what is ndarray in NumPy**    
509. **How would you convert a Pandas DataFrame into a NumPy array?**    
510. **What are the advantages of NumPy over regular Python lists?**    
511. **What are the differences between np.mean() vs np.average() in Python NumPy?**    
512. **What are the differences between NumPy arrays and matrices?**    
513. **What is the difference between Vectorisation vs Broadcasting in NumPy?**    
514. **Explain what is Vectorization in NumPy**    
515. **What is the difference between flatten and ravel functions in NumPy?**    
516. **What does einsum do in NumPy?**    
517. **What is the most efficient way to map a function over a NumPy array?**    
518. **What is the purpose of meshgrid in Python/NumPy?  Related To: Python**  
519. **What is broadcasting in connection to Linear Algebra?  Related To: Linear Algebra**  
520. **Is there a difference between Numpy var() and Pandas var()?  Related To: Pandas**  
521. **How to normalize an array in NumPy to a Unit Vector?**    
522. **What's the difference between a View and a Shallow Copy of a NumPy array?**    
523. **What are Strides in NumPy? How does it work?**    
524. **What is the difference between the following assignment methods?**    
525. **What is the difference between contiguous and non-contiguous arrays?**    
526. **What are some main reason why NumPy is so fast?**    
527. **What is Fortran contiguous arrays?**    
528. **Calculate the Euclidean Distance between two points**     
529. **How to access the i-th column of a NumPy multidimensional array?**    
530. **How to find all occurrences of an Element in a list  Related To: Python**  
531. **How would you reverse a NumPy array?**    
532. **What is the difference between test\[:,0\] vs test\[:,\[0\]\]? When would you use one?**    
533. **Extract all numbers between a given range from a NumPy array**    
534. **What's the easiest way to implement a moving average with NumPy?**    
535. **How to find all the local maxima (or peaks) in a 1D array?**    
536. **Convert array of indices to One-Hot encoded NumPy array**    
537. **How to convert a numeric array to a categorical (text) array?**    
538. **How would you get indices of N-max values in a NumPy array?**    
539. **Transpose matrix using einsum similar to np.transpose(arr)**    
540. **How to convert an array of arrays into a flat 1D array?**    
541. **Write the einsum equivalent of inner, outer, sum, and multiplication functions**    
542. **Consider an array Z \= \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R \= \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]?**    
543. **Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]))**    
544. **Sum over last two axis at once**    
545. **How would you detect Overfitting in Linear Models?  Related To: Linear Regression**  
546. **What is Underfitting in Machine Learning?  Related To: Machine Learning**  
547. **What is Hyper-Parameters in ML Model?  Related To: Machine Learning**  
548. **What is Overfitting in Machine Learning?  Related To: Machine Learning**  
549. **What is a model Learning Rate? Is a high learning rate always good?  Related To: Machine Learning**  
550. **How to know whether your model is suffering from the problem of Exploding Gradients?  Related To: Deep Learning, Neural Networks**  
551. **What methods of Hyperparameters Tuning do you know?  Related To: Optimisation, ML Design Patterns, Dimensionality Reduction**  
552. **What are the difference between Type I and Type II errors?  Related To: Machine Learning, Data Processing**  
553. **How does ROC curve and AUC value help measure how good a model is?  Related To: Classification**  
554. **What performance parameters can be calculated using Confusion Matrix?**    
555. **How would you prevent Overfitting when designing an Artificial Neural Network?  Related To: Neural Networks**  
556. **How would you fix Logistic Regression Overfitting problem?  Related To: Linear Regression, Cost Function**  
557. **What is a Confusion Matrix?  Related To: Supervised Learning, Classification**  
558. **What are some advantages and disadvantages of using AUC to measure the performance of the model?  Related To: Classification**  
559. **What is the F-Score?  Related To: Statistics, Classification**  
560. **How do you reduce the risk of making a Type I and Type II error?  Related To: Statistics**  
561. **How is AUC \- ROC curve used in classification problems?   Related To: Classification**  
562. **How would you use a Confusion Matrix for determining a model performance?  Related To: Data Processing, Classification**  
563. **If one algorithm has Higher Precision but Lower Recall than other, how can you tell which algorithm is better?**    
564. **What is BIC?**    
565. **How would you deal with Overfitting in Linear Regression models?  Related To: Linear Regression**  
566. **How would you choose an evaluation metric for an Imbalanced classification?  Related To: Classification**  
567. **What is AIC?  Related To: Classification**  
568. **What is MDL?**    
569. **What are Concordance and Discordance?**    
570. **Compare AIC and BIC methods for model selection**     
571. **What are some approaches to get a quantitative estimate of a model's Maximum Predictive Power given a certain level of noise?  Related To: Data Processing**  
572. **When Overfitting can be useful?  Related To: ML Design Patterns**  
573. **Why would you need to use the Continued Model Evaluation Design Pattern?  Related To: ML Design Patterns**  
574. **What's the difference between ROC and Precision-Recall Curves?  Related To: Classification**  
575. **How to interpret F-measure values?  Related To: Classification**  
576. **Are there any troubles when using Early Stopping?  Related To: ML Design Patterns, Data Processing, Neural Networks**  
577. **What is Optimisation in Machine Learning?**    
578. **What are Non-Differentiable Objective Functions?**    
579. **What are Bayesian Optimization Methods?**    
580. **Why do you need to know Convex Optimization?**    
581. **What are Differentiable Objective Functions?**    
582. **What methods of Hyperparameters Tuning do you know?  Related To: ML Design Patterns, Model Evaluation, Dimensionality Reduction**  
583. **When would you use Grid Search vs Random Search for Hyperparameter Tuning?  Related To: ML Design Patterns, Dimensionality Reduction**  
584. **What is Random Search Optimization?**    
585. **How does the Adam method of Stochastic Gradient Descent work?  Related To: Gradient Descent**  
586. **How does Batch Size affect the Convergence of SGD and why?  Related To: Gradient Descent**  
587. **When are Genetic Algorithms a good choice for Optimisation?  Related To: Genetic Algorithms**  
588. **How is AdaGrad used to optimize a Learning Rate?**    
589. **When should we use Algorithms like Adam as opposed to SGD?  Related To: Gradient Descent**  
590. **Compare Batch Gradient Descent and Stochastic Gradient Descent  Related To: Gradient Descent**  
591. **When you are Optimizing a Neural Network, does it make sense to combine both Momentum and Weight Decay to improve the performance?  Related To: Neural Networks**  
592. **When Optimizing a Neural Network, how do you define the Termination Condition for Gradient Descent?  Related To: Gradient Descent**  
593. **Why is Newton's Method not used in Optimisation as opposed to Gradient Descent?**    
594. **Why does Newton's Method only use First and Second Derivatives, not Third or Higher Derivatives?**    
595. **How do you use the Convex Optimization Approach to Minimize Regret?**    
596. **Compare using Newton's Method and Gradient Descent for Optimisation**    
597. **What is the difference between Momentum based Gradient Descent and Nesterov's Accelerated Gradient Descent?  Related To: Gradient Descent**  
598. **In what situations would you prefer Coordinate Descent over Gradient Descent?  Related To: Gradient Descent**  
599. **What is the Mirror Descent?  Related To: Gradient Descent**  
600. **Explain in detail how Momentum-based Gradient Descent and Nesterov's Accelerated Gradient Descent work  Related To: Gradient Descent**  
601. **In Random Forests, how do you optimize the Number of Trees T in the Forest?  Related To: Random Forest**  
602. **Should Training Samples Randomly Drawn for Mini-Batch Training Neural Networks be drawn with or without Replacement?  Related To: Neural Networks**  
603. **Why is Newton's Method not sensitive to ill-conditioned Hessian?**    
604. **In Hyperparameter Optimization, would you use Random Search or Grid Search to achieve a better performance?**    
605. **When you are Optimizing your Neural Network, is it a good idea to Prune the Network?  Related To: Neural Networks**  
606. **For both Convex, and Non-Convex Problems, does the Gradient in SGD always point to the global extreme value?  Related To: Gradient Descent**  
607. **How Bayesian Optimisation is used in Hyperparameter Tuning?**    
608. **What does "almost all local minimum have very similar function value to the global optimum" mean?**    
609. **What is an Optimization Strategy which can be used to solve Convex Problems but is Blind to the Problem Structure?**    
610.  **How Principal Component Analysis (PCA) is used for Dimensionality Reduction?  Related To: Dimension Reduction, Unsupervised Learning**  
611. **What is Principal Component Analysis (PCA)?  Related To: Dimensionality Reduction**  
612. **Can we use PCA for feature selection?  Related To: Feature Engineering**  
613. **How is the first principal component axis selected in PCA?  Related To: Dimensionality Reduction**  
614. **Why is Centering and Scaling the data important before performing PCA?  Related To: Linear Algebra, Dimensionality Reduction**  
615. **What is the difference between PCA and Random Projection approaches?  Related To: Dimensionality Reduction**  
616. **What are some advantages of using LLE over PCA?  Related To: Unsupervised Learning, Dimensionality Reduction**  
617. **How do you perform Principal Component Analysis (PCA)?  Related To: Feature Engineering**  
618. **Would you use PCA on large datasets or there is a better alternative?  Related To: Scikit-Learn**  
619. **What's the difference between PCA and t-SNE?  Related To: Dimensionality Reduction**  
620. **Is PCA checks what characteristics are redundant and discards them?  Related To: Dimension Reduction**  
621. **What is Sparse PCA?  Related To: Dimensionality Reduction**  
622. **How is PCA used for Anomaly Detection?  Related To: Anomaly Detection, Unsupervised Learning**  
623. **When would you use Manifold Learning techniques over PCA?  Related To: Dimensionality Reduction**  
624. **What's the difference between Principal Component Analysis and Independent Component Analysis?  Related To: Feature Engineering**  
625. **What is the relationship between k-Means Clustering and PCA?  Related To: Clustering, K-Means Clustering**  
626. **How can you obtain the principal components and the eigenvalues from Scikit-Learn PCA?  Related To: Scikit-Learn**  
627. **How to create new columns derived from existing columns in Pandas?**    
628. **How are iloc() and loc() different?**    
629. **What are the operations that Pandas Groupby method is based on ?**    
630. **Describe how you will get the names of columns of a DataFrame in Pandas**    
631. **In Pandas, what do you understand as a bar plot and how can you generate a bar plot visualization**    
632. **How would you iterate over rows in a DataFrame in Pandas?**    
633. **How to check whether a Pandas DataFrame is empty?**    
634. **What does the in operator do in Pandas?**    
635. **How does the groupby() method works in Pandas?**    
636. **Why do should make a copy of a DataFrame in Pandas?**    
637. **Name some methods you know to replace NaN values of a DataFrame in Pandas**    
638. **Define the different ways a DataFrame can be created in Pandas**    
639. **Name the advantage of using applymap() vs apply() method**    
640. **Describe how you can combine (merge) data on Common Columns or Indices?**    
641. **Compare the Pandas methods: map(), applymap(), apply()**    
642. **When cleaning data, mention how you will identify outliers present in a DataFrame object**    
643. **What is the difference between join() and merge() in Pandas?**    
644. **What is the difference(s) between merge() and concat() in Pandas?**    
645. **When to use merge() over concat() and vice-versa in Pandas?**    
646. **Is it a good idea to iterate over DataFrame rows in Pandas?**    
647. **Name some type conversion methods in Pandas**    
648. **How will you write DataFrame to PostgreSQL table?**    
649. **How can I achieve the equivalents of SQL's IN and NOT IN in Pandas?**    
650. **How would you create Test (20%) and Train (80%) Datasets with Pandas?  Related To: Scikit-Learn**  
651. **Is there a difference between Numpy var() and Pandas var()?  Related To: NumPy**  
652. **How would you convert continuous values into discrete values in Pandas?**    
653. **When would you use Scikit-Learn OneHotEncoder() vs Pandas pd.get\_dummies()?   Related To: Scikit-Learn**  
654. **What's the difference between pivot\_table() and groupby()?**    
655. **What's the difference between interpolate() and fillna() in Pandas?**    
656. **What's the difference between at and iat in Pandas?**    
657. **What are some best-practices to work with Large Files in Pandas?**    
658. **Explain what is Multi-indexing in Pandas?**    
659. **What are some best practises to optimize Pandas code?**    
660. **What is Vectorization in a context of using Pandas?**    
661. **How would you deal with large CSV files in Pandas?**    
662. **What's the difference between apply and transform on a Group object?**     
663. **What does the stack() and unstack() functions do in a DataFrame?**    
664. **How is a Pandas crosstab different from a Pandas pivot\_table? When would you use each one?**    
665. **What is the pipe method? When you would use it?**    
666. **How do you construct a MultiIndex for a DataFrame? Provide an example**    
667. **What is a Probability Distribution?**    
668. **What is the difference between the Bernoulli and Binomial distribution?**    
669. **What's the difference between Probability Mass Functions and Density Probability Functions?**    
670. **What is the difference between a Combination and a Permutation?**    
671. **Find a probability of dangerous Fire when there is Smoke  Related To: Naïve Bayes**  
672. **What is a Poisson process?**    
673. **What's the difference between Disjoint Events and Independent Events?**    
674. **What's the difference between the likelihood and the posterior probability in Bayesian statistics?  Related To: Naïve Bayes**  
675. **When is an event Independent of Itself?**    
676. **What's the difference between Binomial Distribution and Geometric Distribution?**    
677. **How would you test hypotheses using Bayes' Rule?  Related To: Naïve Bayes**  
678. **Name some Probability Distributions you know**    
679. **What is the Bayesian approach to probability?**    
680. **What's the difference between Cumulative Distribution Functions and Probability Density Functions?**    
681. **How do you transform a Skewed Distribution into a Normal Distribution?**    
682. **How would you Calibrate Probabilities for a classification model?  Related To: Classification**  
683. **Name some methods you will use to estimate the Parameters of a Probability Distribution**    
684. **How would you check if two events are Independent?**    
685. **Explain if the inference using the Frequentist Approach will always yield the same result as the Bayesian approach?**    
686. **What is the Cumulative Distribution Function?**    
687. **Why would you use Probability Calibration?  Related To: Classification**  
688. **What is the difference between Bayesian Estimation and Maximum Likelihood Estimation?**    
689. **What's the difference between Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) and in which cases would use each one?  Related To: CNN, Neural Networks**  
690. **What are the uses of using RNN in NLP?  Related To: NLP, Neural Networks**  
691. **Why are RNNs (Recurrent Neural Network) better than MLPs at predicting Time Series Data?  Related To: Time Series**  
692. **What's the difference between Recurrent Neural Networks and Recursive Neural Networks?  Related To: Neural Networks**  
693. **How many dimensions must the inputs of an RNN layer have? What does each dimension represent? What about its outputs?  Related To: Neural Networks**  
694. **Why would you use Encoder-Decoder RNNs vs plain sequence-to-sequence RNNs for automatic translation?  Related To: NLP, Neural Networks**  
695. **What's the difference between Traditional Feedforward Networks and Recurrent Neural Networks?  Related To: Neural Networks**  
696. **What are the main difficulties when training RNNs? How can you handle them?  Related To: Neural Networks**  
697. **What types of Recurrent Neural Networks (RNN) do you know?  Related To: Neural Networks**  
698. **What's the difference between Stateful RNN vs Stateless RNN? What are their pros and cons?  Related To: NLP, Neural Networks**  
699. **When would you use MLP, CNN, and RNN?  Related To: CNN, Neural Networks**  
700. **Explain the intuition behind RNN having a Vanishing Gradient Problem?  Related To: Neural Networks**  
701. **Compare Feed-forward and Recurrent Neural Network  Related To: Neural Networks**  
702. **How to calculate the output of a Recurrent Neural Network (RNN)?  Related To: Neural Networks**  
703. **Explain how a Recurrent Architecture for leveraging visual attention works  Related To: Neural Networks**  
704. **How does LSTM compare to RNN?  Related To: Neural Networks**  
705. **Why would you want to use 1D convolutional layers in an RNN?  Related To: Neural Networks**  
706. **How would you define Random Forest?  Related To: Ensemble Learning**  
707. **Explain how the Random Forests give output for Classification, and Regression problems?**    
708. **Does Random Forest need Pruning? Why or why not?**    
709. **What are Ensemble Methods?  Related To: Ensemble Learning**  
710. **How is a Random Forest related to Decision Trees?  Related To: Decision Trees, Ensemble Learning**  
711. **What are some hyperparameters in Random Forest?**    
712. **How would you find the optimal size of the Bootstrapped Dataset?**    
713. **Is it necessary to do Cross Validation in Random Forest?**    
714. **Is Random Forest an Ensemble Algorithm?  Related To: Ensemble Learning**  
715. **How do you determine the Depth of the Individual Trees?**    
716. **How does Random Forest handle missing values?**    
717. **What is Entropy criteria used to split a node?**    
718. **What is Variable Selection and what are its Objectives in Random Forest?**    
719. **How would you improve the performance of Random Forest?  Related To: Feature Engineering**  
720. **How is it possible to perform Unsupervised Learning with Random Forest?  Related To: Unsupervised Learning**  
721. **Why Random Forest models are considered not interpretable?**    
722. **What is Out-of-Bag Error?**    
723. **How does the number of trees affect the Random Forest model?**    
724. **What does Random refer to in Random Forest?**    
725. **When would you use SVMs over Random Forest and vice-versa?  Related To: SVM**  
726. **Explain the advantages of using Random Forest**    
727. **What are some drawbacks of using Random Forest?**    
728. **Explain the concept behind BAGGing  Related To: Ensemble Learning**  
729. **What is the difference between OOB score and validation score?  Related To: Decision Trees, Ensemble Learning**  
730. **What are proximities in Random Forests?**    
731. **How would you define the criteria to split on at each node of the trees?**    
732. **Why is the training efficiency of Random Forest better than Bagging?**    
733. **What is AdaBoost algorithm?**    
734. **What is the Isolation Forest Algorithm?  Related To: Anomaly Detection**  
735. **How are feature\_importances\_ in RandomForestClassifier determined in Scikit-Learn?  Related To: Scikit-Learn**  
736. **How can you tell the importance of features using Random Forest?**    
737. **What is Gini Impurity used to split a node?**    
738. **Imagine that you know there are outliers in your data, would you use Logistic Regression?  Related To: Anomaly Detection, Decision Trees, Logistic Regression**  
739. **Give some reasons to choose Random Forests over Neural Networks  Related To: Ensemble Learning, Neural Networks**  
740. **What are some advantages of Neural Network over Random Forest?  Related To: Neural Networks**  
741. **Explain how it is possible to get feature importance in Random Forest using Out Of Bag Error**    
742. **Does Random Forest suffer from the Curse of Dimensionality?  Related To: Dimensionality Reduction, Curse of Dimensionality**  
743. **In Random Forests, how do you optimize the Number of Trees T in the Forest?  Related To: Optimisation**  
744. **How would you find the optimal number of random features to consider at each split?  Related To: Ensemble Learning**  
745. **Explain a method of Variable Selection for Random Forest**    
746. **What technique would you use to prevent Swamping and Masking for Isolation Forest Anomaly Detection?  Related To: Anomaly Detection**  
747. **What are Recommendation Systems?**    
748. **How are Knowledge-based Recommender Systems different from Collaborative and Content-based Recommender Systems?**    
749. **What is the difference between Collaborative and Content based Recommender Systems?**    
750. **What are some Domain-Specific Challenges in Recommender Systems?**    
751. **What are some applications of Recommender Systems?**    
752. **What is a Model-Based Collaborative approach?**    
753. **What are the basic components of a Content-Based System?  Related To: Naïve Bayes**  
754. **What is the difference between Personalization Systems, User-Adaptive Systems, and Recommender Systems?**    
755. **What is the basic structure of a Content-Based Recommender System?**    
756. **How would you create a Recommender System for Text Inputs? PY Related To: Python**  
757. **How do you use Naive Bayes model for Collaborative Filtering?  Related To: Naïve Bayes**  
758. **How do you maintain User Privacy when collecting Data for Recommendation Systems?**    
759. **What are some advantages of using Neighborhood-based approaches for Recommender Systems?**    
760. **How does the Neighborhood-based Recommendation work?**    
761. **What is the difference between Constraint-based Recommender Systems and Case-based Recommender Systems?**    
762. **What is the importance of Multi-Armed Bandit Methods for Computational Advertising?**    
763. **How do you choose between User-Based and Item-Based Neighborhood Recommender System?**    
764. **What are the different types of Memory-Based Collaborative approaches?**    
765. **What are some advantages of Content-Based Recommendation paradigm over Collaborative-Based Recommendation?**    
766. **What are the different methods that you can collect User Data for the Recommendation Process?**    
767. **What are the differences between Computational Advertising and Recommender Systems?**    
768. **How should Recommender Systems work in a Changing Environment?**    
769. **What are the differences between Content-Based and Collaborative Methods in terms of Bias and Variance?  Related To: Bias & Variance**  
770. **What is the difference between Factorisation Machines and Matrix Factorisation?**    
771. **What is the difference between Item-Item Collaborative Filtering and Market Basket Analysis?**    
772. **How can SVD be used in Collaborative Filtering in Theory?**    
773. **What algorithm does Google News Personalisation Engine use?**    
774. **What is the difference between Collaborative Quality Filtering and Collaborative Filtering?**    
775. **How do you extend Decision Trees to Collaborative Filtering?  Related To: Decision Trees**  
776. **Can SVD work for Recommender Systems?**    
777. **What is Support Vector Machine?  Related To: Supervised Learning**  
778. **What is Hyperplane in SVM?**    
779. **What are Support Vectors in SVMs?**    
780. **What are Hard-Margin and Soft-Margin SVMs?**    
781. **What types of SVM kernels do you know?**    
782. **Why would you use the Kernel Trick?  Related To: Classification**  
783. **What are some applications of SVMs?**    
784. **Name some advantages of SVM**    
785. **For N dimensional data set what is the minimum possible number of Support Vectors?**    
786. **What happens when there is no clear Hyperplane in SVM?**    
787. **How can less Training Data give Higher Accuracy?  Related To: Data Processing**  
788. **What is the Hinge Loss in SVM?  Related To: Classification, Cost Function**  
789. **What is the role of C hyperparameter in SVM?**    
790. **While designing an SVM classifier, what values should the designer select?**    
791. **Compare K-Nearest Neighbors (KNN) and SVM  Related To: K-Nearest Neighbors**  
792. **What is the difference between Classification and Regression when using SVM?**    
793. **What are the Convex Hulls?**    
794. **How to use one-class SVM for Anomalies Detections?  Related To: Anomaly Detection**  
795. **Compare SVM and Logistic Regression in handling outliers  Related To: Anomaly Detection, Logistic Regression**  
796. **What is the Kernel Trick?**    
797. **When would you use SVMs over Random Forest and vice-versa?  Related To: Random Forest**  
798. **What are some similarities between SVMs and Neural Networks?  Related To: Neural Networks**  
799. **What are some differences between SVMs and Neural Networks?  Related To: Neural Networks**  
800. **When SVM is not a good approach?**    
801. **What is Ranking SVM?**    
802. **What are Support Vectors?**    
803. **Provide an intuitive explanation of Linear Support Vector Machines (SVMs)  Related To: Classification**  
804. **What is Quadratic Optimisation Problem in SVM?**    
805. **What are Polynomial Kernels?**    
806. **What is the difference between a Decision Boundary and a Hyperplane?**    
807. **Does Redundant data affect an SVM-based classifier?  Related To: Data Processing**  
808. **Why is the Lagrangian important in SVM?**    
809. **What is the Dual Problem?**    
810. **How does the value of Gamma affect the SVM?**    
811. **How does the value of C affect the SVM?**    
812. **Is there a relation between the Number of Support Vectors and the classifiers performance?**    
813. **Explain the dual form of SVM formulation**    
814. **What is Structured SVM?**    
815. **What is Sequential Minimal Optimization?  Related To: Neural Networks**  
816. **Name some advantages of using Support Vector Machines vs Logistic Regression for classification  Related To: Logistic Regression, Classification**  
817. **When would you use SVM vs Logistic regression?  Related To: Logistic Regression, Classification**  
818. **What are Slack Variables in SVM?**    
819. **What are C and Gamma (γ) with regards to a Support Vector Machine?**    
820. **What are Radial Basis Function Kernels?**    
821. **How would you deal with classification on Non-linearly Separable data?  Related To: Classification**  
822. **Can you explain PAC learning theory intuitively?  Related To: Machine Learning**  
823. **What is the difference between Deep Learning and SVM?  Related To: Deep Learning**  
824. **Can Support Vector Machines be used for Outlier Detection?  Related To: Anomaly Detection**  
825. **How does the Algorithm "The 10% You Don't Need" remove the Redundant Data?  Related To: Data Processing**  
826. **Does linear SVMs suffer from the Curse of Dimensionality?  Related To: Dimensionality Reduction, Curse of Dimensionality**  
827. **What is Mercer's theorem and how is it related to SVM?**    
828. **How do you approximate RBF kernel to scale with large numbers of training samples?**    
829. **Why is SVM not popular nowadays? Also, when did SVM perform poorly?**    
830. **Why does SVM work well in practice, even if the reproduced space is very high dimensional?**    
831. **What is the Probably Approximately Correct learning?  Related To: Machine Learning**  
832. **How to select Kernel for SVM?**    
833. **What is Data Leakage and how do you avoid it in Scikit-Learn?**    
834. **How to obtain reproducible results across multiple program executions in Scikit-Learn?**    
835. **What's the difference between GridSearchCV and RandomSearchCV? What is the advantage of each one?**   
836. **What are the basic objects that a ML pipeline should contain?  Related To: Feature Engineering**  
837. **What's the problem if you call the fit() method multiple times with different X and y data? How can you overcome this issue?**    
838. **What's the difference between StandardScaler and Normalizer and when would you use each one?**    
839. **What is the difference between Recursive Feature Elimination (RFE) function and SelectFromModel in Scikit-Learn?**    
840. **How would you create Test (20%) and Train (80%) Datasets with Pandas?  Related To: Pandas**  
841. **While using KNeighborsClassifier, when would you set weights="distance"?**    
842. **How do you optimize the Ridge Regression parameter?**    
843. **When to use OneHotEncoder vs LabelEncoder in Scikit-Learn?  Related To: Data Processing**  
844. **Is max\_depth in Scikit-learn the equivalent of pruning in decision trees? If not, how a decision tree is pruned using scikit?**    
845. **What's the difference between StratifiedKFold (with shuffle \= True) and StratifiedShuffleSplit in Scikit-Learn?  Related To: Data Processing**  
846. **Suppose you have multiple CPU cores available, how can you use them to reduce the computational cost?**    
847. **What's the difference between fit(), transform() and fit\_transform()? Why do we need these separate methods?**    
848. **How do you scale data that has many outliers in Scikit-Learn?**    
849. **When would you use Scikit-Learn OneHotEncoder() vs Pandas pd.get\_dummies()?   Related To: Pandas**  
850. **How would you split Train and Test samples in imbalanced classifications?**    
851. **How are feature\_importances\_ in RandomForestClassifier determined in Scikit-Learn?  Related To: Random Forest**  
852. **What is the difference between cross\_validate and cross\_val\_score in Scikit-Learn?**    
853. **Would you use PCA on large datasets or there is a better alternative?  Related To: PCA**  
854. **Are there any advantage of XGBoost over GradientBoostingClassifier?**    
855. **How does sklearn KNeighborsClassifier compute class probabilites when setting weights='uniform'?**    
856. **Can you use SVM with a custom kernel in Scikit-Learn?**    
857. **Does scikit-learn have a forward selection/stepwise regression algorithm?**    
858. **When you would use StratifiedKFold instead of KFold?**    
859. **How does Recursive Feature Elimination (RFE) works in Scikit-learn?**    
860. **When you would use TheilSenRegressor, RANSAC and HuberRegressor?**    
861. **How to adjust the hyperparameters of MLP classifier using GridSearchCV to get more perfect performance?**    
862. **What is Linear Regression?  Related To: Linear Regression**  
863. **What are Decision Trees?  Related To: Decision Trees**  
864. **What do you understand by the term Supervised Learning?**    
865. **What is Support Vector Machine?  Related To: SVM**  
866. **What is a Perceptron?  Related To: Neural Networks, Classification**  
867. **What are the two types of problems solved by Supervised Learning?**    
868. **What is the difference between Supervised Learning and Unsupervised Learning?  Related to: Unsupervised Learning**  
869. **Why Naive Bayes is called Naive?  Related To: Naïve Bayes, Classification**  
870. **What is the difference between KNN and K-means Clustering?  Related To: K-Means Clustering, Unsupervised Learning, Classification**  
871. **Explain the structure of a Decision Tree  Related To: Decision Trees**  
872. **Define Linear Regression and its structure  Related To: Linear Regression**  
873. **Give a real life example of Supervised Learning and Unsupervised Learning  Related To: Unsupervised Learning**  
874. **What is Bias in Machine Learning?  Related To: Bias & Variance**  
875. **How are the different nodes of decision trees represented?  Related To: Decision Trees**  
876. **What is the Bias-Variance tradeoff?  Related To: Bias & Variance**  
877. **What is the difference between a Regression problem and a Classification problem?**    
878. **What is k-Nearest Neighbors algorithm?  Related To: K-Nearest Neighbors**  
879. **In statistics, what is the difference between Bias and Error?  Related To: Statistics**  
880. **What is Cross-Validation and why is it important in supervised learning?  Related To: Data Processing**  
881. **What is the difference between a Multiclass problem and a Multilabel problem?  Related To: Clustering**  
882. **What is the Bias Error?  Related To: Bias & Variance**  
883. **What is the Variance Error?  Related To: Bias & Variance**  
884. **What are some challenges faced when using a Supervised Regression Model?  Related To: Linear Regression**  
885. **How do you use a supervised Logistic Regression for Classification?  Related To: Logistic Regression, Classification**  
886. **What is a Confusion Matrix?  Related To: Model Evaluation, Classification**  
887. **How is Gradient Boosting used to improve Supervised Learning?  Related To: Ensemble Learning**  
888. **What are some disadvantages of Supervised Learning?**    
889. **What is the difference between Supervised and Unsupervised learning?  Related To: Unsupervised Learning**  
890. **Compare Reinforced Learning and Supervised Learning  Related To: Reinforcement Learning**  
891. **What is the difference between Gradient Boosting and Adaptive Boosting?  Related To: Decision Trees**  
892. **What is Semi-Supervised learning?**    
893. **How do you choose between Supervised and Unsupervised learning?  Related To: Unsupervised Learning**  
894. **What is Time Series?**    
895. **How are CNNs used for Time Series Prediction?  Related To: CNN, Neural Networks**  
896. **What are some common Data Preparation Operations you would use for Time Series Data?**    
897. **How do you handle Missing Values in Time-Series Data?**    
898. **How the IQR (Interquartile Range) is used in Time Series Forecasting?  Related To: Anomaly Detection**  
899. **What is the Sliding Window method for Time Series Forecasting?**    
900. **Why does a Time Series have to be Stationary?**    
901. **What are some real-world applications of Time-Series Forecasting?**    
902. **What does "irregularly spaced spatial data" mean?**    
903. **What are some examples of Time-Series Data which can be Mined?**    
904. **Can Non-Sequential Deep Learning Models outperform Sequential Models in Time-Series Forecasting?**    
905. **If your Time-Series Dataset is very long, what architecture would you use?**    
906. **How is Pearson Correlation used with Time Series?**    
907. **What statistical methods can I use to assess the differences between the time series?**    
908. **List some Advantages of using State-Space Models and Kalman Filter for Time-Series Modelling**    
909. **What is Discrete Wavelet Transform?**    
910. **What would be some reasons that LSTM Models do not improve the Time-Series Forecasting significantly as compared to MLP Models?**    
911. **Why are RNNs (Recurrent Neural Network) better than MLPs at predicting Time Series Data?  Related To: RNN**  
912. **How do you Normalise Time-Series Data?**    
913. **What are some different Neural Architectures for predicting Time-Series Values?**    
914. **What are some advantages of using MLP (Multilayer Perceptrons) for Time Series Prediction?**    
915. **How do you mine Contextual Spatial Attributes?**    
916. **Compare some Forecasting Techniques for Stationary and Non-stationary Time-Series**    
917. **What process do you go through to design a Neural Network for Time-Series Forecasting?**    
918. **How would you compare the two Time Series shown below?**    
919. **What is a Moving Average Process? Give some Real-life Examples.**    
920. **What are some Similarity Measures used for Sequence Data?  Related To: Data Mining**  
921. **What are some Similarity Measures which can be used with Time Series data?**    
922. **How do you Auto-correlate Discrete and Abstract Time Series Data?**    
923. **Explain briefly the different methods of Noise-Removal for Time-Series Data**    
924. **What are some different ways of Trajectory Patterns Mining?**    
925. **Explain how the Facebook Prophet is used to predict Time-Series Data**    
926. **After you develop a Real-Time Classifier of Time Series Events, how do you know if a Low Power Embedded System can classify events in real-time?**    
927. **Can Hidden Markov Models be used to model Time-Series data?**    
928. **Why does Time-Series have to be Stationary before you can run ARIMA or ARM Models?**    
929. **When would you use Sequential Split of data?   Related To: ML Design Patterns, Data Processing**  
930. **Compare State-Space Models and ARIMA models**    
931. **What are some common Machine Learning problems that Unsupervised Learning can help with?  Related To: Feature Engineering**  
932. **What are some applications of Unsupervised Learning?**    
933. **What is the difference between Supervised Learning and Unsupervised Learning?  Related To: Supervised Learning**  
934. **How Principal Component Analysis (PCA) is used for Dimensionality Reduction?  Related To: Dimension Reduction, PCA**  
935. **What is the difference between KNN and K-means Clustering?  Related To: K-Means Clustering, Supervised Learning, Classification**  
936. **Give a real life example of Supervised Learning and Unsupervised Learning  Related To: Supervised Learning**  
937. **What is the Curse of Dimensionality and how can Unsupervised Learning help with it?  Related To: Dimensionality Reduction, Curse of Dimensionality**  
938. **How can Neural Networks be Unsupervised?  Related To: Autoencoders, Neural Networks**  
939. **How is it possible to perform Unsupervised Learning with Random Forest?  Related To: Random Forest**  
940. **How does K-Means perform Clustering?  Related To: K-Means Clustering**  
941. **What is the difference between Supervised and Unsupervised learning?  Related To: Supervised Learning**  
942. **What are some differences between Unsupervised Learning and Reinforcement Learning?**    
943. **Describe the approach used in Denoising Autoencoders  Related To: Autoencoders**  
944. **What is the difference between the two types of Hierarchical Clustering?  Related To: Clustering**  
945. **What are some advantages of using LLE over PCA?  Related To: PCA, Dimensionality Reduction**  
946. **What’s the LDA algorithm? Give an example.**    
947. **What is the difference and connection between Clustering and Dimension Reduction?  Related To: Clustering, Dimensionality Reduction**  
948. **Why does K-Means have a higher bias when compared to Gaussian Mixture Model?  Related To: Clustering**  
949. **Explain the Locally Linear Embedding algorithm for Dimensionality Reduction  Related To: Dimensionality Reduction**  
950. **How do you choose between Supervised and Unsupervised learning?  Related To: Supervised Learning**  
951. **Can you use Batch Normalisation in Sparse Auto-encoders?  Related To: Autoencoders**  
952. **What are the main differences between Sparse Autoencoders and Convolution Autoencoders?  Related To: Autoencoders**  
953. **What are some differences between the Undercomplete Autoencoder and the Sparse Autoencoder?  Related To: Autoencoders**  
954. **How is PCA used for Anomaly Detection?  Related To: Anomaly Detection, PCA**  
955. **Explain how a cluster is formed in the DBSCAN Clustering Algorithm  Related To: Clustering**  
956. **How to tell if data is clustered enough for clustering algorithms to produce meaningful results?  Related To: Clustering, K-Means Clustering**  
957. **Are GANs unsupervised?**    
958. 