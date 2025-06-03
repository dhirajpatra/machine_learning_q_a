# machine_learning_qa
Preapring set for machine learning engineer

# 100 Machine Learning Questions for HackerEarth Test Preparation

## Section 1: Machine Learning Fundamentals (20 Questions)

1. **Q1:** What is the main difference between supervised and unsupervised learning? Provide examples of each.

2. **Q2:** Explain the bias-variance tradeoff. How does it relate to model complexity?

3. **Q3:** What is overfitting and how can you detect it? List 3 techniques to prevent overfitting.

4. **Q4:** Explain k-fold cross-validation. Why is it better than a simple train-test split?

5. **Q5:** What is the difference between Type I and Type II errors? Which one is more critical in medical diagnosis?

6. **Q6:** Calculate precision, recall, and F1-score given: TP=85, FP=15, TN=120, FN=20.

7. **Q7:** When would you use ROC-AUC vs PR-AUC? Explain with examples.

8. **Q8:** What is stratified sampling and when should you use it?

9. **Q9:** Explain the difference between bagging and boosting ensemble methods.

10. **Q10:** What is feature selection? Compare filter, wrapper, and embedded methods.

11. **Q11:** How do you handle missing data? List and explain 4 different approaches.

12. **Q12:** What is the curse of dimensionality? How does it affect machine learning models?

13. **Q13:** Explain the difference between L1 and L2 regularization. When would you use each?

14. **Q14:** What is data leakage? Provide 3 examples of how it can occur.

15. **Q15:** How do you evaluate a regression model? List 5 different metrics.

16. **Q16:** What is the difference between parametric and non-parametric models?

17. **Q17:** Explain hold-out validation, cross-validation, and bootstrap sampling.

18. **Q18:** What is the No Free Lunch theorem in machine learning?

19. **Q19:** How do you handle imbalanced datasets? List 5 techniques.

20. **Q20:** What is the difference between online learning and batch learning?

## Section 2: Classification Algorithms (15 Questions)

21. **Q21:** Derive the logistic regression cost function from maximum likelihood estimation.

22. **Q22:** What are the assumptions of logistic regression? What happens if they're violated?

23. **Q23:** How do decision trees handle categorical and numerical features differently?

24. **Q24:** What is information gain and how is it calculated? Provide the formula.

25. **Q25:** Explain the difference between Gini impurity and entropy for decision tree splitting.

26. **Q26:** How does Random Forest reduce overfitting compared to a single decision tree?

27. **Q27:** What is the optimal hyperplane in SVM? How is it determined?

28. **Q28:** Explain the kernel trick in SVM. When would you use RBF vs polynomial kernels?

29. **Q29:** What is the margin in SVM? How do support vectors relate to it?

30. **Q30:** Explain the naive assumption in Naive Bayes. Why does it still work well in practice?

31. **Q31:** How do you handle zero probabilities in Naive Bayes? What is Laplace smoothing?

32. **Q32:** What is the optimal value of k in k-NN? How do you determine it?

33. **Q33:** How does the choice of distance metric affect k-NN performance?

34. **Q34:** Compare the computational complexity of training vs prediction for different algorithms.

35. **Q35:** When would you choose logistic regression over SVM for binary classification?

## Section 3: Regression Algorithms (10 Questions)

36. **Q36:** Derive the normal equation for linear regression. What are its limitations?

37. **Q37:** What are the assumptions of linear regression? How do you test them?

38. **Q38:** Explain multicollinearity. How do you detect and handle it?

39. **Q39:** How does ridge regression solve the multicollinearity problem?

40. **Q40:** What is the difference between Ridge and Lasso regression? When to use each?

41. **Q41:** What is Elastic Net regression? Write its cost function.

42. **Q42:** How do you interpret coefficients in polynomial regression?

43. **Q43:** What is homoscedasticity? How do you test for it?

44. **Q44:** Explain the concept of regularization path in Ridge/Lasso regression.

45. **Q45:** How do you perform feature selection using Lasso regression?

## Section 4: Clustering Algorithms (10 Questions)

46. **Q46:** How do you choose the optimal number of clusters in k-means? Explain the elbow method.

47. **Q47:** What are the limitations of k-means clustering? When does it fail?

48. **Q48:** Explain the k-means++ initialization method. Why is it better than random initialization?

49. **Q49:** What is the difference between hierarchical and partitional clustering?

50. **Q50:** How do you interpret a dendrogram in hierarchical clustering?

51. **Q51:** Explain the concept of linkage criteria in hierarchical clustering.

52. **Q52:** What is DBSCAN? How does it handle noise and outliers?

53. **Q53:** How do you choose epsilon and min_samples parameters in DBSCAN?

54. **Q54:** Compare k-means, hierarchical clustering, and DBSCAN. When to use each?

55. **Q55:** What is the silhouette score? How do you interpret it?

## Section 5: Dimensionality Reduction (10 Questions)

56. **Q56:** Explain PCA step by step. What are principal components?

57. **Q57:** How do you choose the number of components in PCA? Explain cumulative explained variance.

58. **Q58:** What is the difference between PCA and Linear Discriminant Analysis (LDA)?

59. **Q59:** When would you use t-SNE over PCA? What are t-SNE's limitations?

60. **Q60:** How does t-SNE preserve local structure in high-dimensional data?

61. **Q61:** What is the curse of dimensionality in the context of distance metrics?

62. **Q62:** Explain the difference between feature selection and feature extraction.

63. **Q63:** How do you handle categorical variables before applying PCA?

64. **Q64:** What is kernel PCA? When would you use it?

65. **Q65:** How do you interpret the loading matrix in PCA?

## Section 6: Deep Learning Basics (15 Questions)

66. **Q66:** Explain the forward propagation process in a neural network.

67. **Q67:** Derive the backpropagation algorithm for a simple neural network.

68. **Q68:** What is the vanishing gradient problem? How do you solve it?

69. **Q69:** Compare different activation functions: sigmoid, tanh, ReLU, Leaky ReLU.

70. **Q70:** What is the difference between batch, mini-batch, and stochastic gradient descent?

71. **Q71:** Explain the Adam optimizer. How does it combine momentum and RMSprop?

72. **Q72:** What is dropout? How does it prevent overfitting?

73. **Q73:** Explain batch normalization. Why does it help training?

74. **Q74:** What is the difference between L1 and L2 regularization in neural networks?

75. **Q75:** How do you initialize weights in neural networks? Why is it important?

76. **Q76:** Explain the architecture of a Convolutional Neural Network (CNN).

77. **Q77:** What is max pooling in CNNs? What are its benefits?

78. **Q78:** Explain the concept of receptive field in CNNs.

79. **Q79:** What is the difference between RNN, LSTM, and GRU?

80. **Q80:** How do you handle the exploding gradient problem in RNNs?

## Section 7: Data Preprocessing & Engineering (10 Questions)

81. **Q81:** How do you handle outliers in your dataset? List 5 methods.

82. **Q82:** What is the difference between normalization and standardization? When to use each?

83. **Q83:** Explain one-hot encoding vs label encoding for categorical variables.

84. **Q84:** How do you handle high-cardinality categorical features?

85. **Q85:** What is feature scaling? Why is it important for distance-based algorithms?

86. **Q86:** How do you create polynomial features? What are the risks?

87. **Q87:** Explain the concept of feature hashing (hashing trick).

88. **Q88:** How do you handle datetime features in machine learning?

89. **Q89:** What is target encoding? What are its potential problems?

90. **Q90:** How do you detect and handle duplicate records in your dataset?

## Section 8: Python/Programming (10 Questions)

91. **Q91:** Write Python code to split data into train/validation/test sets with stratification.

92. **Q92:** How do you handle memory issues when working with large datasets in pandas?

93. **Q93:** Write code to perform k-fold cross-validation using scikit-learn.

94. **Q94:** How do you save and load trained models in scikit-learn?

95. **Q95:** Write code to create a confusion matrix and classification report.

96. **Q96:** How do you handle categorical features in scikit-learn pipelines?

97. **Q97:** Write code to perform hyperparameter tuning using GridSearchCV.

98. **Q98:** How do you create custom transformers in scikit-learn?

99. **Q99:** Write code to plot ROC curves for multiple models.

100. **Q100:** How do you handle class imbalance using SMOTE in your ML pipeline?

## Section 9: Probability and Statistics (30 Questions)

### Basic Probability (10 Questions)

101. **Q101:** A coin is flipped 3 times. What is the probability of getting at least 2 heads?

102. **Q102:** In a deck of 52 cards, what is the probability of drawing 2 aces without replacement?

103. **Q103:** Explain the difference between independent and mutually exclusive events with examples.

104. **Q104:** What is conditional probability? Calculate P(A|B) if P(A∩B) = 0.3 and P(B) = 0.6.

105. **Q105:** State and explain Bayes' theorem. Provide a real-world ML application.

106. **Q106:** A diagnostic test is 95% accurate. If 1% of population has a disease, what's the probability a person with positive test actually has the disease?

107. **Q107:** What is the difference between permutation and combination? When do you use each?

108. **Q108:** If X and Y are independent random variables, prove that Var(X + Y) = Var(X) + Var(Y).

109. **Q109:** A box contains 5 red, 3 blue, and 2 green balls. What's the probability of drawing 2 red balls and 1 blue ball in 3 draws without replacement?

110. **Q110:** Explain the birthday paradox. What's the probability that in a group of 23 people, at least 2 share the same birthday?

### Probability Distributions (10 Questions)

111. **Q111:** What is a probability density function (PDF) vs cumulative distribution function (CDF)?

112. **Q112:** A random variable X follows normal distribution N(100, 25). Find P(90 < X < 110).

113. **Q113:** When would you use binomial distribution? What are its parameters and assumptions?

114. **Q114:** Explain the relationship between binomial and normal distributions. When can you use normal approximation?

115. **Q115:** What is Poisson distribution? Give 3 real-world examples where it applies.

116. **Q116:** Compare exponential and Poisson distributions. How are they related?

117. **Q117:** What is the Central Limit Theorem? Why is it important in machine learning?

118. **Q118:** A manufacturing process produces 2% defective items. In a batch of 1000 items, what's the probability of having exactly 25 defective items?

119. **Q119:** Explain the difference between uniform, normal, and exponential distributions in terms of their shapes and applications.

120. **Q120:** What is the law of large numbers? How does it relate to sample size in ML?

### Statistical Inference (10 Questions)

121. **Q121:** What is the difference between population and sample? Define parameter vs statistic.

122. **Q122:** Explain Type I and Type II errors in hypothesis testing. What are their consequences?

123. **Q123:** What is a p-value? How do you interpret a p-value of 0.03?

124. **Q124:** Perform a one-sample t-test: Sample mean = 105, population mean = 100, sample std = 15, n = 25. Test at α = 0.05.

125. **Q125:** What is the difference between one-tailed and two-tailed tests? When do you use each?

126. **Q126:** Explain confidence intervals. What does a 95% confidence interval mean?

127. **Q127:** What is the difference between correlation and causation? Provide examples.

128. **Q128:** Calculate Pearson correlation coefficient for: X = [1,2,3,4,5], Y = [2,4,6,8,10].

129. **Q129:** When would you use chi-square test? What are its assumptions?

130. **Q130:** What is ANOVA? When do you use it instead of t-test?

## Section 10: Statistical Methods in ML (20 Questions)

### Descriptive Statistics (5 Questions)

131. **Q131:** Calculate mean, median, mode, and standard deviation for: [1,2,2,3,4,4,4,5,6,100].

132. **Q132:** What is the difference between sample standard deviation and population standard deviation?

133. **Q133:** Explain quartiles, IQR, and how to detect outliers using the IQR method.

134. **Q134:** What is skewness and kurtosis? How do they affect your choice of ML algorithms?

135. **Q135:** How do you handle skewed distributions in machine learning? List 5 transformation techniques.

### Hypothesis Testing in ML (8 Questions)

136. **Q136:** How do you test if two ML models perform significantly differently? Describe the statistical test.

137. **Q137:** What is A/B testing? How do you determine statistical significance in A/B tests?

138. **Q138:** You have two models with accuracies 0.85 and 0.82 on 1000 test samples each. Are they significantly different?

139. **Q139:** What is the multiple testing problem? How does Bonferroni correction address it?

140. **Q140:** Explain paired t-test vs unpaired t-test. When do you use each in ML model comparison?

141. **Q141:** What is statistical power? How does sample size affect it?

142. **Q142:** How do you test for normality of residuals in regression? List 3 tests.

143. **Q143:** What is heteroscedasticity? How do you test for it and what are its implications?

### Bayesian Statistics (4 Questions)

144. **Q144:** Explain the difference between frequentist and Bayesian approaches to statistics.

145. **Q145:** What is a prior, likelihood, and posterior in Bayesian inference?

146. **Q146:** How does Bayesian inference relate to machine learning? Give examples.

147. **Q147:** What is conjugate prior? Provide an example with beta-binomial conjugacy.

### Experimental Design (3 Questions)

148. **Q148:** What is randomization in experimental design? Why is it important?

149. **Q149:** Explain the concept of confounding variables. How do you control for them?

150. **Q150:** What is the difference between observational studies and controlled experiments in the context of ML?

## Section 11: Practical Statistics Problems (20 Questions)

### Data Analysis Scenarios (10 Questions)

151. **Q151:** You notice your model performs worse on Fridays. How would you statistically test if day-of-week affects performance?

152. **Q152:** A/B test shows 5.2% conversion for variant A (n=10000) vs 5.8% for variant B (n=10000). Is the difference significant?

153. **Q153:** Your dataset has 3 features with correlations: r12=0.8, r13=0.9, r23=0.85. What statistical concerns arise?

154. **Q154:** How would you test if customer ages are normally distributed across different product categories?

155. **Q155:** You have click-through rates from 5 different ad campaigns. How do you test if they're significantly different?

156. **Q156:** A feature has 80% missing values. How do you statistically justify dropping vs imputing it?

157. **Q157:** Your model's residuals show a pattern when plotted against predictions. What statistical tests would you perform?

158. **Q158:** How do you test if the variance of errors is constant across different ranges of predicted values?

159. **Q159:** You suspect your training and test sets come from different distributions. How do you test this statistically?

160. **Q160:** How would you test if adding a new feature significantly improves model performance?

### Real-world Applications (10 Questions)

161. **Q161:** A pharmaceutical company claims their drug works in 70% of cases. In a trial of 100 patients, 62 improved. Test the claim.

162. **Q162:** An e-commerce site wants to test if a new checkout process reduces cart abandonment. Design the statistical test.

163. **Q163:** You're analyzing customer lifetime value. The data is highly skewed. What statistical approaches would you use?

164. **Q164:** A bank wants to know if loan default rates differ significantly across geographical regions. How do you approach this?

165. **Q165:** You're predicting house prices. How do you test if your model's assumptions about error distribution hold?

166. **Q166:** A streaming service notices viewing patterns differ by country. How do you quantify and test these differences?

167. **Q167:** You have sales data with strong seasonality. How do you statistically model and test for seasonal effects?

168. **Q168:** A social media platform wants to test if a new algorithm increases user engagement. Design the experiment.

169. **Q169:** You're analyzing survey data with Likert scales (1-5 ratings). What statistical considerations apply?

170. **Q170:** A delivery company claims average delivery time is 2 days. You sample 50 deliveries with mean 2.3 days, std 0.8 days. Test the claim.

---

## Updated Study Tips:

1. **Practice coding**: For programming questions, implement solutions in Python
2. **Understand concepts**: Don't just memorize formulas, understand the intuition
3. **Real datasets**: Practice with actual datasets from Kaggle or UCI repository
4. **Time management**: Practice solving questions within time limits
5. **Documentation**: Always explain your reasoning and approach
