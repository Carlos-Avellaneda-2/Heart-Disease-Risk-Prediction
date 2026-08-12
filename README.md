# Heart Disease Risk Prediction — Logistic Regression Homework

## Carlos Avellaneda

## Exercise Summary

This project implements logistic regression **from scratch** (NumPy only, no scikit-learn for
model training) to predict the presence of heart disease from clinical patient data. The work
covers the full pipeline requested in the assignment:

- Exploratory data analysis (missing values, duplicates, outlier screening, class balance).
- A hand-built sigmoid function, binary cross-entropy cost, and gradient descent, trained on a
  manually implemented stratified 70/30 train/test split.
- Visualization of linear decision boundaries for three feature pairs (`age`–`chol`,
  `trestbps`–`thalach`, `oldpeak`–`ca`).
- L2 regularization added to the cost and gradient functions, with λ tuned over
  `[0, 0.001, 0.01, 0.1, 1]`.
- Training and testing of the final model in Amazon SageMaker (AWS Academy), using the same
  train/test split and preprocessing produced locally, **without creating or deploying an
  endpoint**.

All steps, metrics, plots, and interpretation are in [`heart_disease_lr_analysis.ipynb`](./heart_disease_lr_analysis.ipynb).

## Dataset Description

- **Name:** Heart Disease Dataset (UCI Cleveland collection, Kaggle mirror).
- **Source:** https://www.kaggle.com/datasets/neurocipher/heartdisease
- **Records:** 303 patients, 14 clinical attributes plus the binary target
  (`1` = disease present, `0` = disease absent).
- **Features used for modeling (6, as required by the assignment):** `age`, `chol` (serum
  cholesterol), `trestbps` (resting blood pressure), `thalach` (maximum heart rate achieved),
  `oldpeak` (ST depression induced by exercise), and `ca` (number of major vessels colored by
  fluoroscopy).

> To reproduce the notebook locally: download the CSV from the Kaggle link above and place it in
> the same folder as `heart_disease_lr_analysis.ipynb` before running it — the notebook
> auto-detects the CSV file in its working directory.

## SageMaker Evidence

The notebook prepares `train_sagemaker.csv`, `test_sagemaker.csv`, and
`preprocessing_params.csv` for upload to the AWS Academy SageMaker environment (see Section 14 of
the notebook). **No endpoint was created or deployed** — SageMaker was used strictly to train and
test the model, per the account restriction stated in the assignment.

Add your screenshots below as evidence that training and testing were completed successfully in
SageMaker.

### 1. SageMaker notebook / training execution

*Screenshot showing the SageMaker notebook instance running the training code
(e.g. the notebook cell that calls the training loop, with the kernel and instance visible).*

```
![SageMaker training execution](./screenshots/sagemaker_training.png)
```

### 2. Successful completion

*Screenshot showing the training run finished without errors (e.g. final cost/loss printed,
"Training complete" message, or the completed cell with its output).*

```
![SageMaker training completed](./screenshots/sagemaker_completion.png)
```

### 3. Test-set metrics

*Screenshot showing accuracy, precision, recall, and F1 computed on the held-out test set inside
the SageMaker notebook.*

```
![SageMaker test metrics](./screenshots/sagemaker_test_metrics.png)
```

### Environment / instance configuration

*Record here the SageMaker notebook instance type / kernel used (e.g. `ml.t3.medium`,
`conda_python3`), if available from the AWS Academy environment.*

-

### Local vs. SageMaker comparison

*After running SageMaker, copy the metrics into Section 16 of the notebook
(`heart_disease_lr_analysis.ipynb`) to replace the `NaN` placeholders, and summarize here whether
the results matched the local run.*

-

> **Reminder:** do not include endpoint configuration or inference-endpoint evidence — only
> training and testing screenshots are required, per the assignment's SageMaker account
> restriction.

## Repository Contents

- `heart_disease_lr_analysis.ipynb` — end-to-end notebook (EDA, model implementation,
  visualization, regularization, SageMaker data preparation).
- `README.md` — this file.
- `heart.csv` — dataset used to run the notebook (or place your own downloaded copy here).
- `screenshots/` — SageMaker evidence images (add this folder with your screenshots before
  submitting).
