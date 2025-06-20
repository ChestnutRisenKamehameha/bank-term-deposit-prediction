# xgboost-bankmarketing

# Bank Marketing Term Deposit Prediction

This project uses machine learning to predict whether a client will subscribe to a term deposit based on data from a Portuguese bank's marketing campaigns.

---

## Goal

Predict if a client will **subscribe to a term deposit**  
Target variable: `y` → `"yes"` / `"no"`

---

## Dataset Information

- **Source:** `bank-full.csv`
- **Size:** ~45,000 records
- **Features:** 17 input features

### Features Overview

#### Client Attributes
- `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`

#### Campaign Attributes
- `contact`, `month`, `day`, `duration`, `campaign`

#### Economic Indicators
- `pdays`, `previous`, `poutcome`

#### Target
- `y` (binary): `"yes"` or `"no"`

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

---

## How It Works

1. **Loading and Preprocessing Data**
2. **Transforming Target to Binary** → `y_bin`
3. **One-hot Encoding** for categorical variables
4. **Spliting** into train and test sets
5. Training three models:
   - Decision Tree
   - Random Forest
   - XGBoost Classifier
6. **Evaluating** models using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
7. **Visualizing** model comparison

---

## Results

A bar chart compares the performance of the models:

- **Decision Tree**
- **Random Forest**
- **XGBoost**

## Output Sample

![Model Comparison](./comparison_chart.png)

---

## How to Run

```bash
git clone https://github.com/yourusername/bank-term-deposit-prediction.git
cd bank-term-deposit-prediction
pip install -r requirements.txt
python predict.py
