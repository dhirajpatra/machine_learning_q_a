# Complete Answer Keys for 170 ML Questions

## Section 1: Machine Learning Fundamentals (Answers 1-20)

**A1:** Supervised learning uses labeled data to learn input-output mappings (e.g., classification, regression). Unsupervised learning finds patterns in unlabeled data (e.g., clustering, dimensionality reduction). Examples: Supervised - email spam detection, house price prediction; Unsupervised - customer segmentation, anomaly detection.

**A2:** Bias-variance tradeoff describes the balance between model simplicity and complexity. High bias (underfitting) = overly simple models miss patterns. High variance (overfitting) = complex models are sensitive to training data noise. Optimal complexity minimizes total error = bias² + variance + irreducible error.

**A3:** Overfitting occurs when models memorize training data but fail on new data. Detection: Large gap between training and validation performance. Prevention techniques: (1) Cross-validation, (2) Regularization (L1/L2), (3) Early stopping, (4) Dropout, (5) Data augmentation.

**A4:** K-fold cross-validation splits data into k folds, trains on k-1 folds, tests on 1 fold, repeats k times. Benefits over train-test split: (1) Uses all data for both training and testing, (2) Provides more robust performance estimates, (3) Reduces variance in performance metrics, (4) Better for small datasets.

**A5:** Type I error (False Positive): Rejecting true null hypothesis. Type II error (False Negative): Accepting false null hypothesis. In medical diagnosis, Type II is more critical - missing a disease (false negative) can be life-threatening, while false alarms (Type I) cause unnecessary anxiety but less harm.

**A6:** Given: TP=85, FP=15, TN=120, FN=20
- Precision = TP/(TP+FP) = 85/(85+15) = 0.85
- Recall = TP/(TP+FN) = 85/(85+20) = 0.81
- F1-score = 2×(Precision×Recall)/(Precision+Recall) = 2×(0.85×0.81)/(0.85+0.81) = 0.83

**A7:** ROC-AUC for balanced datasets; measures ability to distinguish classes across all thresholds. PR-AUC for imbalanced datasets; focuses on positive class performance. Use PR-AUC when positive class is rare (fraud detection, medical diagnosis) as ROC-AUC can be overly optimistic.

**A8:** Stratified sampling maintains the same proportion of samples from each class/group as in the original population. Use when: (1) Dataset is imbalanced, (2) Ensuring representative samples, (3) Maintaining class distribution in train/test splits, (4) Survey sampling across demographics.

**A9:** Bagging (Bootstrap Aggregating): Trains multiple models on different bootstrap samples, reduces variance, models trained in parallel (Random Forest). Boosting: Trains models sequentially, each corrects previous model's errors, reduces bias (AdaBoost, Gradient Boosting).

**A10:** Feature selection chooses relevant features from existing ones. Filter methods: Statistical tests (chi-square, correlation). Wrapper methods: Use model performance (forward/backward selection, RFE). Embedded methods: Built into algorithm (L1 regularization, tree feature importance).

**A11:** Missing data handling: (1) Deletion - listwise/pairwise removal, (2) Mean/median/mode imputation, (3) Forward/backward fill for time series, (4) Interpolation methods, (5) Model-based imputation (KNN, regression), (6) Multiple imputation, (7) Use algorithms that handle missing values (XGBoost).

**A12:** Curse of dimensionality: As dimensions increase, data becomes sparse, distances become meaningless, computational complexity increases exponentially. Effects: (1) Increased sample size requirements, (2) Distance-based algorithms fail, (3) Overfitting risk increases, (4) Visualization becomes impossible.

**A13:** L1 (Lasso): Penalty = λΣ|βᵢ|, creates sparse solutions, performs feature selection. L2 (Ridge): Penalty = λΣβᵢ², shrinks coefficients uniformly, handles multicollinearity. Use L1 for feature selection, L2 for regularization without sparsity.

**A14:** Data leakage: Future information inadvertently used to predict past events. Examples: (1) Using future data in features, (2) Target leakage - features derived from target, (3) Temporal leakage - shuffling time series data, (4) Preprocessing before splitting data.

**A15:** Regression metrics: (1) MAE - Mean Absolute Error, (2) MSE - Mean Squared Error, (3) RMSE - Root Mean Squared Error, (4) R² - Coefficient of determination, (5) MAPE - Mean Absolute Percentage Error, (6) Adjusted R², (7) Huber loss.

**A16:** Parametric models assume specific functional form with fixed parameters (linear regression, logistic regression). Non-parametric models make no assumptions about underlying distribution, adapt to data (k-NN, decision trees, kernel methods).

**A17:** Hold-out: Single train-test split. Cross-validation: Multiple train-test splits, average performance. Bootstrap: Sampling with replacement, estimates sampling distribution. Cross-validation most reliable for model evaluation, bootstrap for confidence intervals.

**A18:** No Free Lunch theorem states no algorithm performs best on all problems. Implies: (1) Need domain knowledge for algorithm selection, (2) No universal best algorithm, (3) Must match algorithm to problem characteristics, (4) Importance of empirical testing.

**A19:** Imbalanced datasets techniques: (1) Resampling - SMOTE, undersampling, (2) Cost-sensitive learning, (3) Ensemble methods, (4) Different evaluation metrics (PR-AUC), (5) Threshold tuning, (6) Anomaly detection approaches.

**A20:** Batch learning: Trains on entire dataset at once, requires retraining for new data. Online learning: Updates model incrementally with new data points, adapts continuously. Online learning better for streaming data, concept drift, large datasets.

## Section 2: Classification Algorithms (Answers 21-35)

**A21:** Logistic regression cost function derivation:
Given sigmoid: p = 1/(1+e^(-z)), z = βᵀx
Likelihood: L = Πpᵢʸⁱ(1-pᵢ)^(1-yᵢ)
Log-likelihood: ℓ = Σ[yᵢlog(pᵢ) + (1-yᵢ)log(1-pᵢ)]
Cost function: J = -ℓ = -Σ[yᵢlog(pᵢ) + (1-yᵢ)log(1-pᵢ)]

**A22:** Logistic regression assumptions: (1) Linear relationship between logit and features, (2) Independence of observations, (3) No multicollinearity, (4) Large sample size. Violations lead to: Biased coefficients, poor predictions, unstable results.

**A23:** Decision trees handle categorical features by testing membership in subsets. For numerical features, they find optimal split points by testing thresholds. Categorical splits can be binary or multi-way, numerical splits are always binary (≤ threshold).

**A24:** Information Gain = Entropy(parent) - Weighted_Average_Entropy(children)
Entropy = -Σpᵢlog₂(pᵢ)
Example: If parent has entropy 1.0 and children have weighted average entropy 0.6, Information Gain = 0.4

**A25:** Gini Impurity = 1 - Σpᵢ²; ranges 0-0.5 for binary classification. Entropy = -Σpᵢlog₂(pᵢ); ranges 0-1 for binary. Both measure node purity. Gini is faster to compute, entropy gives more balanced trees.

**A26:** Random Forest reduces overfitting through: (1) Bootstrap sampling - different training sets, (2) Random feature selection at each split, (3) Averaging predictions from multiple trees, (4) Each tree sees different data perspectives, (5) Ensemble reduces variance.

**A27:** Optimal hyperplane in SVM maximally separates classes with largest margin. Determined by: (1) Finding support vectors (closest points to boundary), (2) Maximizing distance between support vectors, (3) Solving quadratic optimization problem, (4) Decision boundary equidistant from closest points of each class.

**A28:** Kernel trick maps data to higher dimensions without explicit computation. RBF kernel: Good for non-linear, complex boundaries, local influence. Polynomial kernel: Good for global patterns, interactions between features. Choose based on data complexity and computational constraints.

**A29:** Margin is distance between hyperplane and closest data points. Support vectors are training points that lie on margin boundaries. They define the hyperplane - only support vectors matter for decision boundary, other points can be removed without changing the model.

**A30:** Naive assumption: Features are conditionally independent given class. Works well because: (1) Violating independence doesn't always hurt classification, (2) Correlation structure may be similar across classes, (3) Simple model reduces overfitting, (4) Works well even with moderate dependence.

**A31:** Zero probabilities handled by Laplace smoothing: Add α (usually 1) to all counts.
P(feature|class) = (count + α) / (total_count + α × num_features)
Prevents zero probabilities that would make entire prediction zero.

**A32:** Optimal k determined by: (1) Cross-validation testing different k values, (2) Odd k for binary classification to avoid ties, (3) k = √n rule of thumb, (4) Typically k between 3-20. Too small k → overfitting, too large k → underfitting.

**A33:** Distance metrics affect k-NN performance:
- Euclidean: Good for continuous features
- Manhattan: Better for high dimensions, outlier robust
- Hamming: For categorical features
- Cosine: For text/sparse data
Feature scaling crucial for distance-based metrics.

**A34:** Training complexity: k-NN O(1), Decision Tree O(n log n), SVM O(n³), Naive Bayes O(nd)
Prediction complexity: k-NN O(nd), Decision Tree O(log n), SVM O(sv×d), Naive Bayes O(cd)
Where n=samples, d=features, sv=support vectors, c=classes

**A35:** Choose Logistic Regression when: (1) Need probability estimates, (2) Linear decision boundary sufficient, (3) Interpretable coefficients needed, (4) Fast training/prediction required. Choose SVM when: (1) Non-linear boundaries needed, (2) High-dimensional data, (3) Robust to outliers required.

## Section 3: Regression Algorithms (Answers 36-45)

**A36:** Normal equation: β = (XᵀX)⁻¹Xᵀy
Limitations: (1) Requires XᵀX to be invertible, (2) Computationally expensive O(n³), (3) Doesn't work with regularization, (4) Fails with multicollinearity, (5) Memory intensive for large datasets.

**A37:** Linear regression assumptions: (1) Linearity, (2) Independence, (3) Homoscedasticity, (4) Normality of residuals. Tests: (1) Scatter plots for linearity, (2) Durbin-Watson for independence, (3) Breusch-Pagan for homoscedasticity, (4) Shapiro-Wilk for normality.

**A38:** Multicollinearity: High correlation between predictors. Detection: (1) VIF > 10, (2) Correlation matrix, (3) Condition number > 30. Solutions: (1) Remove correlated features, (2) Ridge regression, (3) PCA, (4) Combine correlated features.

**A39:** Ridge regression adds L2 penalty: J = MSE + λΣβᵢ²
Shrinks coefficients toward zero but doesn't eliminate them. Handles multicollinearity by stabilizing coefficient estimates when XᵀX is near-singular.

**A40:** Ridge (L2): Shrinks coefficients uniformly, handles multicollinearity, keeps all features. Lasso (L1): Performs feature selection, creates sparse solutions, can eliminate features. Use Ridge for prediction with all features, Lasso for feature selection.

**A41:** Elastic Net combines L1 and L2 penalties:
J = MSE + λ₁Σ|βᵢ| + λ₂Σβᵢ²
Cost function = MSE + α×ρ×L1 + α×(1-ρ)×L2
Where α controls overall regularization, ρ balances L1/L2.

**A42:** Polynomial regression coefficients represent feature interactions. For y = β₀ + β₁x + β₂x²: β₁ is linear effect, β₂ is quadratic effect. Higher-order terms capture non-linear relationships but interpretation becomes complex with interactions.

**A43:** Homoscedasticity: Constant variance of residuals across all predicted values. Test using: (1) Breusch-Pagan test, (2) White test, (3) Residual plots. Violation (heteroscedasticity) leads to inefficient estimates and invalid confidence intervals.

**A44:** Regularization path shows how coefficients change with penalty parameter λ. As λ increases: Ridge coefficients shrink smoothly to zero, Lasso coefficients drop to exactly zero at different λ values, creating feature selection path.

**A45:** Lasso performs automatic feature selection by setting coefficients to exactly zero. Features with non-zero coefficients at optimal λ are selected. Cross-validation determines optimal λ, resulting subset of features has non-zero coefficients.

## Section 4: Clustering Algorithms (Answers 46-55)

**A46:** Elbow method plots within-cluster sum of squares (WCSS) vs number of clusters. Optimal k is at the "elbow" where WCSS reduction slows dramatically. Other methods: Silhouette analysis, Gap statistic, AIC/BIC for model selection.

**A47:** K-means limitations: (1) Must specify k beforehand, (2) Assumes spherical clusters, (3) Sensitive to initialization, (4) Affected by outliers, (5) Struggles with varying cluster sizes/densities, (6) Only finds convex clusters.

**A48:** K-means++ initialization chooses centers to maximize distance from existing centers:
1. Choose first center randomly
2. For each subsequent center, choose with probability proportional to squared distance from nearest existing center
3. Reduces iterations needed, gives better final clustering

**A49:** Hierarchical clustering builds tree of clusters, either agglomerative (bottom-up) or divisive (top-down). Partitional clustering directly divides data into k clusters. Hierarchical shows cluster relationships, partitional is more efficient for large datasets.

**A50:** Dendrogram shows hierarchical clustering as tree structure. Height represents distance between merged clusters. Horizontal cuts at different heights give different numbers of clusters. Longer branches indicate more distinct clusters.

**A51:** Linkage criteria determine distance between clusters:
- Single: Minimum distance between any two points
- Complete: Maximum distance between any two points  
- Average: Average distance between all pairs
- Ward: Minimizes within-cluster variance

**A52:** DBSCAN (Density-Based Spatial Clustering) groups dense regions separated by sparse regions. Handles noise by labeling sparse points as outliers. Finds arbitrary-shaped clusters, doesn't require specifying number of clusters.

**A53:** Epsilon (ε): Maximum distance for points to be neighbors. Min_samples: Minimum points needed to form dense region. Choose using: k-distance plot for ε, domain knowledge for min_samples. Typically min_samples = 2×dimensions.

**A54:** 
- K-means: Fast, spherical clusters, requires k specification
- Hierarchical: Shows cluster relationships, computationally expensive O(n³)  
- DBSCAN: Arbitrary shapes, handles noise, sensitive to parameters
Use k-means for spherical clusters, hierarchical for cluster relationships, DBSCAN for arbitrary shapes.

**A55:** Silhouette score measures how well points fit their clusters vs other clusters. Range [-1, 1]: 1 = perfect clustering, 0 = overlapping clusters, -1 = wrong clustering. Average silhouette score evaluates overall clustering quality.

## Section 5: Dimensionality Reduction (Answers 56-65)

**A56:** PCA steps:
1. Standardize data
2. Compute covariance matrix
3. Find eigenvalues/eigenvectors
4. Sort by eigenvalues (descending)
5. Select top k eigenvectors (principal components)
6. Transform data: Y = XW
Principal components are orthogonal directions of maximum variance.

**A57:** Choose components using cumulative explained variance ratio. Common thresholds: 80-95% of total variance. Plot explained variance vs component number, look for elbow. Kaiser criterion: Keep components with eigenvalues > 1.

**A58:** PCA finds directions of maximum variance (unsupervised). LDA finds directions that maximize class separation (supervised). PCA preserves total variance, LDA maximizes between-class variance relative to within-class variance.

**A59:** Use t-SNE for visualization of high-dimensional data, preserves local structure better than PCA. t-SNE limitations: (1) Computationally expensive, (2) Non-deterministic, (3) Perplexity parameter sensitive, (4) Only for visualization, not preprocessing.

**A60:** t-SNE converts high-dimensional Euclidean distances to probabilities, then finds low-dimensional embedding that matches these probabilities. Uses Student's t-distribution in low dimensions to avoid crowding problem.

**A61:** In high dimensions, distances between all points become similar (concentration of measure). Nearest and farthest neighbors have similar distances, making distance-based algorithms ineffective. Volume of high-dimensional spaces concentrates in thin shells.

**A62:** Feature selection: Choose subset of original features. Feature extraction: Create new features as combinations of original features. Selection maintains interpretability, extraction can capture complex relationships but loses interpretability.

**A63:** One-hot encode categorical variables before PCA, or use categorical PCA variants. Standard PCA assumes continuous variables. Alternative: Use Multiple Correspondence Analysis (MCA) for categorical data.

**A64:** Kernel PCA applies kernel trick to perform PCA in higher-dimensional space without explicit mapping. Useful for non-linear dimensionality reduction. Common kernels: RBF, polynomial. More computationally expensive than linear PCA.

**A65:** Loading matrix shows correlation between original features and principal components. High absolute loadings indicate which features contribute most to each component. Helps interpret what each principal component represents.

## Section 6: Deep Learning Basics (Answers 66-80)

**A66:** Forward propagation:
1. Input layer receives features
2. For each hidden layer: z = Wx + b, a = activation(z)
3. Output layer: ŷ = final_activation(Wz + b)
4. Compute loss: L = loss_function(y, ŷ)
Information flows forward through network to generate predictions.

**A67:** Backpropagation computes gradients using chain rule:
1. Compute output layer error: δᴸ = ∇ₐL ⊙ σ'(zᴸ)
2. Backpropagate error: δˡ = ((Wˡ⁺¹)ᵀδˡ⁺¹) ⊙ σ'(zˡ)
3. Compute gradients: ∇wL = δˡ(aˡ⁻¹)ᵀ, ∇bL = δˡ
4. Update weights: W := W - α∇wL

**A68:** Vanishing gradients occur when gradients become exponentially small in deep networks. Solutions: (1) ReLU activation, (2) Residual connections, (3) Batch normalization, (4) LSTM/GRU for RNNs, (5) Gradient clipping, (6) Better initialization.

**A69:** Activation functions:
- Sigmoid: σ(x) = 1/(1+e⁻ˣ), range (0,1), suffers vanishing gradients
- Tanh: tanh(x), range (-1,1), zero-centered, still vanishing gradients  
- ReLU: max(0,x), computationally efficient, can die
- Leaky ReLU: max(0.01x,x), prevents dying ReLU problem

**A70:** 
- Batch GD: Uses entire dataset, stable but slow
- Mini-batch GD: Uses subset of data, balance of stability and speed
- Stochastic GD: Uses single sample, fast but noisy
Mini-batch most commonly used in practice.

**A71:** Adam optimizer combines momentum and RMSprop:
- Momentum: Accumulates exponentially decaying moving average of gradients
- RMSprop: Adapts learning rate based on exponentially decaying average of squared gradients
- Adam: Uses both first and second moments with bias correction

**A72:** Dropout randomly sets fraction of neurons to zero during training. Prevents overfitting by: (1) Reducing co-adaptation between neurons, (2) Ensemble effect, (3) Forces network to not rely on specific neurons, (4) Regularization effect.

**A73:** Batch normalization normalizes inputs to each layer: BN(x) = γ(x-μ)/σ + β
Benefits: (1) Reduces internal covariate shift, (2) Allows higher learning rates, (3) Acts as regularizer, (4) Reduces dependence on initialization, (5) Accelerates training.

**A74:** L1 regularization in neural networks adds penalty λΣ|wᵢ| to loss function, promotes sparsity. L2 regularization adds penalty λΣwᵢ², prevents large weights. L2 more common in neural networks, often called weight decay.

**A75:** Weight initialization crucial for training stability:
- Xavier/Glorot: Var(W) = 1/nᵢₙ or 2/(nᵢₙ + nₒᵤₜ)
- He initialization: Var(W) = 2/nᵢₙ (for ReLU)
- Zero initialization causes symmetry problem
- Too large causes exploding gradients, too small causes vanishing gradients

**A76:** CNN architecture:
- Convolutional layers: Apply filters to detect local features
- Pooling layers: Reduce spatial dimensions, provide translation invariance
- Fully connected layers: Final classification/regression
- Typical pattern: Conv→ReLU→Pool→Conv→ReLU→Pool→FC→Output

**A77:** Max pooling takes maximum value from each pooling window. Benefits: (1) Reduces spatial dimensions, (2) Provides translation invariance, (3) Reduces overfitting, (4) Computationally efficient, (5) Retains strongest features.

**A78:** Receptive field is region of input that affects a particular feature map element. Deeper layers have larger receptive fields, capturing more global features. Size depends on kernel sizes and number of layers.

**A79:** 
- RNN: Basic recurrent connection, suffers vanishing gradients
- LSTM: Uses gates (forget, input, output) to control information flow, handles long sequences
- GRU: Simplified LSTM with fewer parameters, similar performance
LSTM/GRU solve vanishing gradient problem in RNNs.

**A80:** Exploding gradients in RNNs handled by: (1) Gradient clipping - cap gradient norm, (2) Proper weight initialization, (3) LSTM/GRU architectures, (4) Batch normalization, (5) Shorter sequence lengths.

## Section 7: Data Preprocessing & Engineering (Answers 81-90)

**A81:** Outlier handling methods:
1. IQR method: Remove points outside Q1-1.5×IQR to Q3+1.5×IQR
2. Z-score: Remove points with |z| > 3
3. Isolation Forest: Anomaly detection algorithm
4. Winsorization: Cap extreme values at percentiles
5. Transformation: Log, Box-Cox to reduce impact

**A82:** Normalization scales to [0,1]: x' = (x-min)/(max-min). Standardization scales to mean=0, std=1: x' = (x-μ)/σ. Use normalization for bounded features, standardization for normally distributed features or when need zero mean.

**A83:** One-hot encoding creates binary columns for each category, suitable for nominal data. Label encoding assigns integers to categories, suitable for ordinal data or tree-based algorithms. One-hot prevents false ordering assumptions.

**A84:** High-cardinality categorical features (many unique values) handled by:
1. Target encoding: Replace category with target mean
2. Frequency encoding: Replace with category frequency  
3. Feature hashing: Hash categories to fixed number of dimensions
4. Grouping: Combine rare categories into "Other"
5. Embedding layers in deep learning

**A85:** Feature scaling ensures all features contribute equally to distance calculations. Important for: k-NN, SVM, neural networks, PCA, clustering. Not needed for tree-based algorithms. Methods: standardization, normalization, robust scaling.

**A86:** Polynomial features create interactions: [x₁, x₂] → [1, x₁, x₂, x₁², x₁x₂, x₂²]. Risks: (1) Exponential feature growth, (2) Overfitting, (3) Multicollinearity, (4) Computational complexity. Use regularization to control complexity.

**A87:** Feature hashing (hashing trick) maps high-dimensional categorical features to lower-dimensional space using hash function. Benefits: (1) Fixed memory footprint, (2) Handles unseen categories, (3) Fast computation. Drawback: Hash collisions cause information loss.

**A88:** Datetime feature engineering:
- Extract components: year, month, day, hour, minute, weekday
- Cyclical encoding: sin/cos for periodic features
- Time differences: days since reference date
- Binning: hour of day → time period
- Lag features for time series

**A89:** Target encoding replaces categorical values with target mean for that category. Problems: (1) Overfitting, (2) Data leakage if not done properly. Solutions: (1) Cross-validation encoding, (2) Add smoothing/regularization, (3) Leave-one-out encoding.

**A90:** Duplicate detection and handling:
1. Exact duplicates: Use pandas.drop_duplicates()
2. Fuzzy duplicates: String similarity metrics, clustering
3. Identify key columns for uniqueness
4. Consider domain knowledge for what constitutes duplicates
5. Document removal decisions

## Section 8: Python/Programming (Answers 91-100)

**A91:** Train/validation/test split with stratification:
```python
from sklearn.model_selection import train_test_split
# First split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
```

**A92:** Large dataset memory management:
1. Use chunking: pd.read_csv(chunksize=10000)
2. Data types optimization: use category, int32 instead of int64
3. Select only needed columns
4. Use Dask for larger-than-memory datasets
5. Process in batches
6. Use generators for iteration

**A93:** K-fold cross-validation:
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

**A94:** Model persistence:
```python
import joblib
# Save model
joblib.dump(model, 'model.pkl')
# Load model
loaded_model = joblib.load('model.pkl')

# Alternative with pickle
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

**A95:** Confusion matrix and classification report:
```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

# Classification report
print(classification_report(y_true, y_pred))
```

**A96:** Categorical features in pipelines:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

**A97:** Hyperparameter tuning with GridSearchCV:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1]
}

grid_search = GridSearchCV(
    SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

**A98:** Custom transformers:
```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Custom transformation logic
        return X_transformed
```

**A99:** ROC curves for multiple models:
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

**A100:** Handling class imbalance with SMOTE:
```python
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# Create pipeline with SMOTE
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Note: Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

## Section 9: Probability and Statistics (Answers 101-130)

### Basic Probability (Answers 101-110)

**A101:** P(at least 2 heads in 3 flips) = P(exactly 2) + P(exactly 3)
P(exactly 2) = C(3,2) × (0.5)² × (0.5)¹ = 3 × 0.125 = 0.375
P(exactly 3) = C(3,3) × (0.5)³ × (0.5)⁰ = 1 × 0.125 = 0.125
Total = 0.375 + 0.125 = 0.5

**A102:** P(2 aces without replacement):
P(1st ace) = 4/52, P(2nd ace|1st ace) = 3/51
P(2 aces) = (4/52) × (3/51) = 12/2652 = 1/221 ≈ 0.0045

**A103:** Independent events: P(A∩B) = P(A)×P(B), occurrence of one doesn't affect the other (coin flips). Mutually exclusive events: P(A∩B) = 0, cannot occur simultaneously (rolling 1 and 6 on single die). Note: Independent events are never mutually exclusive unless one has probability 0.

**A104:** Conditional probability P(A|B) = P(A∩B)/P(B)
Given: P(A∩B) = 0.3, P(B) = 0.6
P(A|B) = 0.3/0.6 = 0.5

**A105:** Bayes' theorem: P(A|B) = P(B|A)×P(A)/P(B)
ML application: Naive Bayes classifier uses Bayes' theorem to calculate P(class|features) = P(features|class)×P(class)/P(features)

**A106:** Let D = disease, T = positive test
Given: P(T|D) = 0.95, P(T|¬D) = 0.05, P(D) = 0.01
P(T) = P(T|D)×P(D) + P(T|¬D)×P(¬D) = 0.95×0.01 + 0.05×0.99 = 0.0095 + 0.0495 = 0.059
P(D|T) = P(T|D)×P(D)/P(T) = (0.95×0.01)/0.059 ≈ 0.161 or 16.1%

**A107:** Permutation: Order matters, P(n,r) = n!/(n-r)!. Use for arrangements (seating order).
Combination: Order doesn't matter, C(n,r) = n!/[r!(n-r)!]. Use for selections (choosing team members).

**A108:** For independent X, Y: Cov(X,Y) = 0
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y) = Var(X) + Var(Y) + 2(0) = Var(X) + Var(Y)

**A109:** Total balls = 10, drawing 3 without replacement
P(2 red, 1 blue) = [C(5,2) × C(3,1) × C(2,0)] / C(10,3)
= (10 × 3 × 1) / 120 = 30/120 = 0.25

**A110:** Birthday paradox: P(at least 2 share birthday) = 1 - P(all different)
P(all different) = (365/365) × (364/365) × ... × (343/365) ≈ 0.493  
P(at least 2 same) = 1 - 0.493 = 0.507 or about 50.7%

### Probability Distributions (Answers 111-120)

**A111:** PDF gives probability density at each point for continuous variables (area under curve = probability). CDF gives cumulative probability P(X ≤ x), always non-decreasing from 0 to 1. For discrete variables, PMF gives exact probabilities.

**A112:** X ~ N(100, 25), so μ = 100, σ = 5
P(90 < X < 110) = P(-2 < Z < 2) where Z = (X-100)/5
Using standard normal table: P(-2 < Z < 2) ≈ 0.9545 or 95.45%

**A113:** Binomial distribution models number of successes in n independent trials with success probability p. Parameters: n (trials), p (success probability). Assumptions: (1) Fixed number of trials, (2) Independent trials, (3) Constant probability, (4) Two outcomes only.

**A114:** As n increases and p is moderate, binomial approaches normal distribution (Central Limit Theorem). Rule of thumb: np ≥ 5 and n(1-p) ≥ 5. Approximation: Binomial(n,p) ≈ Normal(np, np(1-p)).

**A115:** Poisson models rare events in fixed time/space intervals. Examples: (1) Number of emails per hour, (2) Defects per manufactured item, (3) Customer arrivals per minute. Parameter λ = average rate.

**A116:** Exponential models time between Poisson events. If events follow Poisson(λ), then time between events follows Exponential(λ). Exponential is continuous, Poisson is discrete. Both have memoryless property.

**A117:** CLT states that sample means approach normal distribution as sample size increases, regardless of population distribution. Important because: (1) Enables inference with any distribution, (2) Justifies normal approximations, (3) Foundation for confidence intervals and hypothesis tests.

**A118:** Use Poisson approximation: n = 1000, p = 0.02, λ = np = 20
P(X = 25) = e^(-20) × 20^25 / 25! ≈ 0.0446 or 4.46%

**A119:** 
- Uniform: Constant probability over interval, rectangular shape, models complete uncertainty
- Normal: Bell-shaped, symmetric, models many natural phenomena
- Exponential: Decreasing curve, models time between events, has memoryless property

**A120:** Law of Large Numbers: Sample average converges to population mean as n → ∞. In ML: Larger samples give better estimates of population parameters, reducing sampling error and improving model generalization.

### Statistical Inference (Answers 121-130)

**A121:** Population: Complete group of interest. Sample: Subset of population.
Parameter: Numerical summary of population (μ, σ). Statistic: Numerical summary of sample (x̄, s).

**A122:** Type I error (α): Rejecting true null hypothesis (false positive). Type II error (β): Accepting false null hypothesis (false negative). Consequences: Type I leads to unnecessary actions, Type II misses real effects. Power = 1 - β.

**A123:** P-value: Probability of observing test statistic as extreme or more extreme, assuming null hypothesis is true. P-value of 0.03 means 3% chance of observing this result if null hypothesis were true. If α = 0.05, reject null hypothesis.

**A124:** One-sample t-test:
H₀: μ = 100, H₁: μ ≠ 100
t = (x̄ - μ₀)/(s/√n) = (105 - 100)/(15/√25) = 5/3 = 1.667
df = 24, critical value ≈ ±2.064
Since |1.667| < 2.064, fail to reject H₀.

**A125:** One-tailed test: Tests direction of difference (>, <), more powerful for detecting effect in specific direction. Two-tailed test: Tests any difference (≠), more conservative. Use one-tailed when theory predicts direction, two-tailed when testing for any difference.

**A126:** Confidence interval: Range of plausible values for population parameter. 95% CI means if we repeated sampling many times, 95% of intervals would contain true parameter. It's about the procedure, not individual intervals.

**A127:** Correlation: Statistical association between variables. Causation: One variable directly influences another. Correlation doesn't imply causation due to: (1) Confounding variables, (2) Reverse causation, (3) Spurious relationships. Example: Ice cream sales correlate with drowning deaths (confounded by temperature).

**A128:** Pearson correlation: r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ - x̄)²Σ(yᵢ - ȳ)²]
X = [1,2,3,4,5], Y = [2,4,6,8,10]
Perfect linear relationship, so r = 1.0

**A129:** Chi-square test examines association between categorical variables or goodness of fit. Use when: (1) Both variables categorical, (2) Testing independence, (3) Goodness of fit testing. Assumptions: (1) Independent observations, (2) Expected frequencies ≥ 5.

**A130:** ANOVA tests equality of means across multiple groups. Use instead of multiple t-tests to control Type I error rate. Compares between-group variance to within-group variance. F-statistic = MS_between/MS_within.

## Section 10: Statistical Methods in ML (Answers 131-150)

### Descriptive Statistics (Answers 131-135)

**A131:** Data: [1,2,2,3,4,4,4,5,6,100]
- Mean = (1+2+2+3+4+4+4+5+6+100)/10 = 13.1
- Median = (4+4)/2 = 4 (middle values)
- Mode = 4 (appears 3 times)
- Standard deviation ≈ 29.8 (high due to outlier 100)

**A132:** Sample std dev uses (n-1) denominator (Bessel's correction) to correct for bias when estimating population variance. Population std dev uses n denominator when you have entire population. Sample version gives unbiased estimate of population variance.

**A133:** Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1
Outliers: Below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
Robust method unaffected by extreme values, commonly used in boxplots.

**A134:** Skewness measures asymmetry: Positive = right tail longer, Negative = left tail longer. Kurtosis measures tail heaviness. Affects ML: (1) May need transformations, (2) Affects algorithm assumptions, (3) Outlier sensitivity, (4) Choose robust algorithms for skewed data.

**A135:** Transform skewed distributions:
1. Log transformation: log(x) for right-skewed
2. Square root: √x for mild right skew
3. Box-Cox: (x^λ - 1)/λ for optimal λ
4. Yeo-Johnson: Handles negative values
5. Quantile transformation: Maps to uniform/normal distribution

### Hypothesis Testing in ML (Answers 136-143)

**A136:** Compare ML models using paired t-test or McNemar's test:
- Paired t-test: For continuous metrics (RMSE, accuracy) on same test set
- McNemar's test: For comparing classification errors on same samples
- Cross-validated paired t-test: Accounts for multiple CV folds
- Corrected resampled t-test: Adjusts for correlation in CV results

**A137:** A/B testing compares two versions (control vs treatment). Steps:
1. Define hypothesis and success metric
2. Calculate required sample size (power analysis)  
3. Randomize users to groups
4. Run test for predetermined duration
5. Statistical test: Two-sample z-test for proportions
Statistical significance: p-value < α (typically 0.05)

**A138:** Two-proportion z-test:
p₁ = 0.85, p₂ = 0.82, n₁ = n₂ = 1000
Pooled proportion: p̂ = (850 + 820)/2000 = 0.835
SE = √[p̂(1-p̂)(1/n₁ + 1/n₂)] = √[0.835×0.165×0.002] ≈ 0.0166
z = (0.85 - 0.82)/0.0166 ≈ 1.81
p-value ≈ 0.07 > 0.05, not significantly different

**A139:** Multiple testing problem: Increased Type I error rate when performing many tests. If α = 0.05 for each test, probability of at least one false positive increases with number of tests. Bonferroni correction: Use α/k where k = number of tests, controls family-wise error rate.

**A140:** Paired t-test: Same subjects tested twice (before/after treatment), tests mean difference. Unpaired t-test: Different subjects in each group, tests difference in group means. In ML: Use paired for same dataset with different models, unpaired for different datasets.

**A141:** Statistical power: Probability of correctly rejecting false null hypothesis (1 - β). Larger sample size increases power by reducing standard error, making it easier to detect true effects. Power analysis determines required sample size for desired effect detection.

**A142:** Test normality of residuals:
1. Shapiro-Wilk test: Tests if sample comes from normal distribution
2. Kolmogorov-Smirnov test: Compares to normal distribution
3. Anderson-Darling test: More sensitive to tail differences
Visual methods: Q-Q plots, histograms of residuals

**A143:** Heteroscedasticity: Non-constant variance of residuals. Tests:
1. Breusch-Pagan test: Regress squared residuals on predictors
2. White test: More general, includes cross-products
3. Goldfeld-Quandt test: Compare variances of subgroups
Implications: Inefficient estimates, invalid confidence intervals. Solutions: Robust standard errors, weighted least squares.

### Bayesian Statistics (Answers 144-147)

**A144:** Frequentist: Probability as long-run frequency, parameters are fixed but unknown, uses p-values and confidence intervals. Bayesian: Probability as degree of belief, parameters are random variables with distributions, updates beliefs with data using Bayes' theorem.

**A145:** 
- Prior: Initial belief about parameter before seeing data
- Likelihood: Probability of data given parameter value  
- Posterior: Updated belief after seeing data
Bayes' theorem: Posterior ∝ Likelihood × Prior

**A146:** Bayesian ML applications:
1. Bayesian neural networks: Uncertainty quantification
2. Gaussian processes: Probabilistic modeling
3. Bayesian optimization: Hyperparameter tuning
4. Online learning: Update beliefs with new data
5. A/B testing: Bayesian hypothesis testing

**A147:** Conjugate prior: When prior and posterior have same distributional form. Beta-binomial example: If prior is Beta(α,β) and likelihood is Binomial, then posterior is Beta(α+successes, β+failures). Simplifies computation, provides analytical solutions.

### Experimental Design (Answers 148-150)

**A148:** Randomization ensures each unit has equal chance of receiving any treatment. Importance: (1) Eliminates selection bias, (2) Balances known/unknown confounders, (3) Validates statistical inference, (4) Creates comparable groups.

**A149:** Confounding variables are associated with both treatment and outcome, creating spurious associations. Control methods: (1) Randomization, (2) Matching, (3) Stratification, (4) Regression adjustment, (5) Instrumental variables.

**A150:** Observational studies: Researcher observes without manipulating variables, can show association but not causation. Controlled experiments: Researcher manipulates variables, can establish causation. In ML: Most studies are observational, randomized experiments needed for causal claims.

## Section 11: Practical Statistics Problems (Answers 151-170)

### Data Analysis Scenarios (Answers 151-160)

**A151:** Test day-of-week effect using ANOVA or chi-square test:
1. Collect model performance by day of week
2. ANOVA: Test if mean performance differs across days
3. Chi-square: Test if error distribution differs by day
4. Post-hoc tests to identify which days differ
5. Consider temporal patterns and business cycles

**A152:** Two-proportion z-test:
p₁ = 0.052, p₂ = 0.058, n₁ = n₂ = 10000
Pooled p̂ = (520 + 580)/20000 = 0.055
SE = √[0.055×0.945×(1/10000 + 1/10000)] ≈ 0.00323
z = (0.058 - 0.052)/0.00323 ≈ 1.86
p-value ≈ 0.063 > 0.05, marginally not significant

**A153:** High correlations indicate multicollinearity concerns:
1. Check VIF (Variance Inflation Factor) > 10
2. Condition number of correlation matrix > 30
3. May cause unstable coefficients, poor interpretability
4. Solutions: Remove correlated features, use regularization, PCA

**A154:** Test normality across categories:
1. Shapiro-Wilk test within each category
2. Kolmogorov-Smirnov test for each group
3. Visual inspection: Q-Q plots by category
4. Levene's test for equal variances across groups
5. Consider transformations if non-normal

**A155:** Compare multiple campaign CTRs:
1. Chi-square test of independence (campaigns vs clicks)
2. One-way ANOVA on CTR values
3. Post-hoc tests (Tukey HSD) for pairwise comparisons
4. Adjust for multiple comparisons
5. Effect size measures (Cohen's w, eta-squared)

**A156:** Missing data analysis:
1. Missing Completely at Random (MCAR) test
2. Pattern analysis: Are missingness patterns informative?
3. Correlation between missingness and other variables
4. Domain knowledge about missing mechanism
5. Cost-benefit analysis of imputation vs deletion

**A157:** Residual pattern analysis:
1. Plot residuals vs fitted values
2. Look for heteroscedasticity, non-linearity
3. Breusch-Pagan test for heteroscedasticity
4. Durbin-Watson test for autocorrelation
5. Consider polynomial terms, interactions, transformations

**A158:** Test homoscedasticity (constant variance):
1. Breusch-Pagan test: Regress |residuals| on predictors
2. White test: More general heteroscedasticity test
3. Goldfeld-Quandt test: Split data, compare variances
4. Visual: Scale-location plot of residuals
5. Solutions: Robust standard errors, weighted regression

**A159:** Test distribution differences:
1. Two-sample Kolmogorov-Smirnov test
2. Anderson-Darling test for distribution comparison
3. Population Stability Index (PSI) for feature drift
4. Chi-square test for categorical distributions
5. Quantile-quantile (Q-Q) plots for visual comparison

**A160:** Test feature importance:
1. Likelihood ratio test: Compare models with/without feature
2. Permutation importance: Shuffle feature, measure performance drop
3. ANOVA F-test for regression coefficients
4. Cross-validated performance comparison
5. Information criteria (AIC, BIC) for model comparison

### Real-world Applications (Answers 161-170)

**A161:** One-sample proportion test:
H₀: p = 0.70, H₁: p ≠ 0.70
Sample proportion: p̂ = 62/100 = 0.62
SE = √[p₀(1-p₀)/n] = √[0.70×0.30/100] = 0.0458
z = (0.62 - 0.70)/0.0458 = -1.75
p-value ≈ 0.08 > 0.05, fail to reject H₀ at α = 0.05

**A162:** A/B test design for cart abandonment:
1. Hypothesis: New checkout reduces abandonment rate
2. Primary metric: Cart abandonment rate
3. Sample size calculation based on current rate, desired effect size
4. Randomization: Users randomly assigned to old/new checkout
5. Duration: Run until reaching statistical significance and practical significance
6. Analysis: Two-proportion z-test

**A163:** Highly skewed CLV analysis:
1. Log transformation: log(CLV + 1)
2. Robust statistics: Median, IQR instead of mean, SD
3. Bootstrap confidence intervals
4. Quantile regression instead of linear regression
5. Consider mixture models or survival analysis
6. Outlier treatment before analysis

**A164:** Test regional loan default differences:
1. Chi-square test of independence (region vs default status)
2. Logistic regression with region as predictor
3. ANOVA on default rates by region
4. Post-hoc tests for pairwise regional comparisons
5. Control for confounders (income, credit score, etc.)
6. Effect size measures

**A165:** Test regression assumptions for house prices:
1. Linearity: Residuals vs fitted values plot
2. Independence: Durbin-Watson test, check spatial correlation
3. Homoscedasticity: Breusch-Pagan test
4. Normality: Shapiro-Wilk test on residuals
5. No multicollinearity: VIF values < 10
6. Solutions: Transformations, robust regression, GAMs

**A166:** Quantify country viewing differences:
1. ANOVA: Test if mean viewing time differs by country
2. Chi-square: Test if viewing category preferences differ
3. Kruskal-Wallis: Non-parametric alternative to ANOVA
4. Effect sizes: Eta-squared, Cohen's d for pairwise comparisons
5. Multiple comparison corrections
6. Cultural and demographic controls

**A167:** Statistical modeling of seasonality:
1. Decompose time series: Trend, seasonal, residual components
2. Seasonal dummy variables in regression
3. Fourier terms for periodic patterns
4. SARIMA models for seasonal ARIMA
5. Tests: Seasonal unit root tests, periodogram analysis
6. Cross-validation with time-aware splits

**A168:** Social media engagement experiment:
1. Randomized controlled trial: Users assigned to old/new algorithm
2. Metrics: Engagement rate, session duration, user retention
3. Stratified randomization by user characteristics
4. Statistical tests: t-tests for continuous metrics, chi-square for categorical
5. Multiple testing corrections for multiple metrics
6. Long-term vs short-term effects analysis

**A169:** Likert scale analysis considerations:
1. Treat as ordinal data, not continuous
2. Use non-parametric tests: Mann-Whitney U, Kruskal-Wallis
3. Chi-square for independence testing
4. Median and IQR for central tendency
5. Ordinal logistic regression for modeling
6. Consider underlying continuous construct

**A170:** One-sample t-test for delivery time:
H₀: μ = 2, H₁: μ ≠ 2
Sample: n = 50, x̄ = 2.3, s = 0.8
t = (2.3 - 2)/(0.8/√50) = 0.3/0.113 ≈ 2.65
df = 49, critical value ≈ ±2.01
Since |2.65| > 2.01, reject H₀. Delivery time significantly different from 2 days.

---

## Quick Reference Summary

### Key Statistical Concepts for ML:
1. **Hypothesis Testing**: Framework for making data-driven decisions
2. **Probability Distributions**: Foundation for understanding data and algorithms
3. **Bayesian Thinking**: Updating beliefs with evidence
4. **Experimental Design**: Proper setup for causal inference
5. **Assumption Testing**: Validating model prerequisites

### Common Statistical Tests in ML:
- **t-tests**: Compare means, model performance
- **Chi-square**: Categorical relationships, goodness of fit
- **ANOVA**: Compare multiple groups
- **Correlation tests**: Feature relationships
- **Normality tests**: Check assumptions

### Practical Applications:
- **A/B Testing**: Compare model versions
- **Feature Selection**: Statistical significance of predictors
- **Model Validation**: Performance testing and comparison
- **Data Quality**: Distribution testing and outlier detection
- **Business Metrics**: Statistical significance of improvements

Remember: Statistical significance ≠ Practical significance. Always consider effect size and business impact alongside p-values!
