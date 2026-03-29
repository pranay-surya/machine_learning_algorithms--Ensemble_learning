# Ensemble Learning 
[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)](https://numpy.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)](#-contributing)

A structured implementation of the four core ensemble learning techniques in machine learning, each demonstrated for both regression and classification tasks using scikit-learn and real-world datasets.

---

## What is Ensemble Learning?..

Ensemble learning is a machine learning paradigm where **multiple models (base learners) are combined** to produce a prediction that is more accurate and robust than any individual model alone.

The core idea comes from the **wisdom of crowds** — a diverse group of imperfect models, when combined correctly, can outperform a single expert model.

There are three fundamental strategies for building ensembles:

| Strategy | Core Idea | Examples |
|---|---|---|
| **Averaging / Voting** | Train models independently, average their outputs | Voting Regressor, Voting Classifier |
| **Bagging** | Train models on random subsets of data (with replacement) | Bagging Regressor, Random Forest |
| **Boosting** | Train models sequentially, each correcting the previous one's errors | AdaBoost, Gradient Boosting |

---

## Techniques Covered

### 1. Voting

**How it works:**

Multiple base models are trained **independently and in parallel** on the full dataset. Their predictions are then combined:

- **Regression:** Final prediction = weighted or unweighted average of all model outputs
- **Classification (Hard Voting):** Final prediction = majority class label
- **Classification (Soft Voting):** Final prediction = class with highest average predicted probability

$$\hat{y}_{\text{regression}} = \frac{\sum_{i=1}^{n} w_i \cdot f_i(x)}{\sum_{i=1}^{n} w_i}$$

**When to use it:**
- You have several strong models with different strengths
- Models are diverse (different algorithms or different hyperparameters)
- Quick ensemble without retraining or complex pipelines

**Key hyperparameters:**

| Parameter | Description |
|---|---|
| `estimators` | List of `(name, model)` tuples — the base learners |
| `weights` | Optional list of weights per estimator |
| `voting` | `'hard'` or `'soft'` (classifier only) |

 
---

### 2. Bagging

**How it works:**

**Bagging** stands for **Bootstrap Aggregating**. It trains multiple instances of the **same base learner** on different random subsets of the training data, sampled **with replacement** (bootstrap samples). Predictions are then averaged (regression) or majority-voted (classification).

```
Original Dataset
      |
      |--- Bootstrap Sample 1 --> Model 1 --> Prediction 1
      |--- Bootstrap Sample 2 --> Model 2 --> Prediction 2   -->  Aggregate --> Final Prediction
      |--- Bootstrap Sample 3 --> Model 3 --> Prediction 3
```

Bagging primarily **reduces variance** — it works best with high-variance, low-bias models like unpruned Decision Trees.

**When to use it:**
- Your base model overfits (high variance)
- You want to stabilize unstable models like Decision Trees
- You want out-of-bag (OOB) error estimation without a separate validation set

**Key hyperparameters:**

| Parameter | Description |
|---|---|
| `base_estimator` | The base model to bag (default: Decision Tree) |
| `n_estimators` | Number of models to train |
| `max_samples` | Fraction or number of samples per bootstrap |
| `max_features` | Fraction or number of features per model |
| `bootstrap` | If `True`, samples with replacement |
| `oob_score` | If `True`, estimates generalization error using OOB samples |
 
---

### 3. Boosting

**How it works:**

**Boosting** trains models **sequentially**. Each new model focuses on the errors of the previous one — misclassified samples (or high-residual samples) are given higher importance in the next round.

```
Model 1 --> Errors --> Model 2 (focuses on Model 1's errors) --> Errors --> Model 3 ...
```

Two major variants are covered:

**AdaBoost (Adaptive Boosting):**
- Assigns higher weights to misclassified samples after each round
- Final prediction is a weighted vote of all models

**Gradient Boosting:**
- Fits each new model to the **residuals** (negative gradient of the loss function) of the previous ensemble
- More flexible — supports various loss functions

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where $h_m$ is the new weak learner and $\eta$ is the learning rate.

**When to use it:**
- You need maximum predictive accuracy
- Your data has complex, non-linear patterns
- Bias reduction is the priority (opposite of bagging)

**Key hyperparameters:**

| Parameter | Description |
|---|---|
| `n_estimators` | Number of boosting rounds |
| `learning_rate` | Shrinks each model's contribution — lower = more regularization |
| `max_depth` | Depth of each base tree (Gradient Boosting) |
| `subsample` | Fraction of training data used per round (Gradient Boosting) |
| `loss` | Loss function to optimize (`'squared_error'`, `'absolute_error'`, etc.) |

 

---

### 4. Random Forest

**How it works:**

Random Forest is an extension of Bagging specifically designed for **Decision Trees**. In addition to bootstrap sampling (rows), it also introduces **random feature selection at each split** — only a random subset of features is considered when finding the best split.

This extra randomness **decorrelates** the trees, reducing variance further compared to standard bagging.

```
For each tree in the forest:
  1. Draw a bootstrap sample of training data
  2. At each node split, randomly select sqrt(n_features) features
  3. Find the best split among those features only
  4. Grow the tree to full depth (no pruning)

Final prediction: average (regression) or majority vote (classification)
```

**Why it works:**

Standard bagged trees can still be correlated if a few very strong features dominate every split. By limiting feature choices at each node, Random Forest forces trees to explore different patterns, resulting in a more diverse and stronger ensemble.

**When to use it:**
- Go-to baseline for tabular data
- You need feature importance rankings
- You want a strong model that is relatively robust to hyperparameter choices

**Key hyperparameters:**

| Parameter | Description |
|---|---|
| `n_estimators` | Number of trees in the forest |
| `max_features` | Features considered at each split: `'sqrt'` (classification default), `'log2'`, or a fraction |
| `max_depth` | Maximum depth per tree (default: None — grows fully) |
| `min_samples_split` | Minimum samples required to split a node |
| `min_samples_leaf` | Minimum samples required at a leaf node |
| `oob_score` | Use out-of-bag samples for error estimation |
| `n_jobs` | Parallel jobs (`-1` = use all cores) |


---

## Datasets Used

| Notebook | Dataset | Task | Source |
|---|---|---|---|
| voting_regressor | California Housing | Predict median house value | `sklearn.datasets` |
| voting_classifier | Breast Cancer Wisconsin | Malignant vs Benign | `sklearn.datasets` |
| bagging_regressor | California Housing | Predict median house value | `sklearn.datasets` |
| bagging_classifier | Wine | Classify wine type (3 classes) | `sklearn.datasets` |
| boosting_regressor | California Housing | Predict median house value | `sklearn.datasets` |
| boosting_classifier | Titanic | Survival prediction | Kaggle / seaborn |
| random_forest_regressor | California Housing | Predict median house value | `sklearn.datasets` |
| random_forest_classifier | Iris | Classify flower species | `sklearn.datasets` |

```

**requirements.txt**

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
xgboost

```

## Concept Map


                        ENSEMBLE LEARNING
                               |
          _________________________|_________________________
         |                         |                        |
      VOTING                   BAGGING                  BOOSTING
  (Parallel, diverse          (Parallel,              (Sequential,
    algorithms)            same algorithm,         corrects errors)
         |                 bootstrap data)                  |
   VotingRegressor               |                   AdaBoost
   VotingClassifier          BaggingRegressor        GradientBoosting
                             BaggingClassifier       XGBoost
                                  |
                           RANDOM FOREST
                      (Bagging + random feature
                         selection per split)
                               |
                    RandomForestRegressor
                    RandomForestClassifier

```

---

## When to Use What

| Scenario | Recommended Technique |
|---|---|
| You have multiple good, diverse models | Voting |
| Your model overfits (high variance) | Bagging or Random Forest |
| You need the highest possible accuracy | Boosting (GradientBoosting / XGBoost) |
| You need feature importances quickly | Random Forest |
| Interpretability is a priority | Voting (transparent base models) |
| Training time is limited | Random Forest (parallelizable) |
| Small dataset | Bagging (OOB evaluation saves data) |

---
