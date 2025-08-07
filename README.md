# Welcome to Visa For Lisa
***

## Task
Galaxy Bank wants to upsell existing **deposit customers** to become **personal-loan customers**.  
The challenge is to **accurately predict** which clients are likely to accept the offer so marketing budget is spent only on high-probability prospects, raising conversion above the historical **≈ 9 %** baseline.

## Description
We tackled the problem in the classic five-stage data-science workflow:

1. **Data Collecting / Cleaning**  
   * Loaded `Visa_For_Lisa_Loan_Modelling.csv`.  
   * Removed duplicates, imputed / dropped missing values.  
   * Encoded categorical columns and standardised numeric ones.

2. **Data Exploration**  
   * Generated summary stats and correlation heatmap.  
   * Identified key drivers of acceptance (Income, CCAvg, CD Account, etc.).

3. **Data Visualization**  
   * Plotted acceptance rate by segment, feature distributions, and correlations for executive insight.

4. **Machine Learning**  
   * Built a `StandardScaler → LogisticRegression` pipeline.  
   * Used stratified train/test split (70 / 30) and 5-fold CV.  
   * Achieved ROC-AUC around **0.90±0.02** on validation data.  
   * Produced lift charts to translate model scoring into marketi

5. **Communication**  
   * Created notebook visuals and a slide deck summarising business impact, risks, and recommended targeting threshold.

All code lives in a single reproducible Jupyter notebook (`visa_for_lisa.ipynb`); the trained model artifact is saved as `loan_acceptance_model.joblib`.

## Installation
```bash
# clone the repo
git clone <repo_url>
cd visa_for_lisa

# set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
JupyterLab / Notebook must also be installed (`pip install jupyterlab`).
