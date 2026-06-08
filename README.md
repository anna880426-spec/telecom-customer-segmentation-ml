# Telecom Subscription Prediction

An end-to-end machine learning project predicting whether a customer will subscribe to a term deposit, based on a telecom marketing campaign dataset. The project covers exploratory data analysis, machine learning modelling, business-driven customer segmentation, and full API deployment.

🔗 **Live API:** [https://telecom-customer-segmentation-ml.onrender.com/docs](https://telecom-customer-segmentation-ml.onrender.com/docs)

---

## Project Structure

```
├── tele_EDA_final_github.ipynb            # Exploratory Data Analysis
├── 36103_25P_AT3_Code_final_github.ipynb  # Modelling & Business Analysis
├── main.py                                # FastAPI application
├── Dockerfile                             # Docker configuration
├── requirements.txt                       # Dependencies
├── model.pkl                              # Trained LightGBM model
├── encoder.pkl                            # Fitted OneHotEncoder
├── scaler.pkl                             # Fitted StandardScaler
├── TeleCom_Data-1.csv                     # Raw data (EDA)
├── TeleCom_Data-2.csv                     # Raw data (Modelling)
└── README.md
```

---

## Problem Statement

A telecommunications company launching a new marketing campaign seeks to identify which customer segments respond most effectively to promotional activities, in order to allocate campaign resources efficiently and maximise subscription rates.

---

## Dataset

- **Source:** Telecom marketing campaign data
- **Size:** 41,188 rows × 21 columns (before cleaning)
- **Target variable:** `y` — whether the customer subscribed (1) or not (0)
- **Class imbalance:** ~11% positive class (y=1)

---

## Part 1 — Exploratory Data Analysis

**Notebook:** `tele_EDA_final_github.ipynb`

### Key Steps
- Data overview, duplicate removal, and type conversion
- `pdays = 999` identified as missing value indicator, not "never contacted"
- Distribution analysis across all features
- Subscription rate by feature with 95% confidence intervals
- Correlation heatmap
- Feature Deep Dive: education × age group analysis, 20–30 age group behavioural patterns

### Key Findings
- **High-value segment:** customers aged 20–30 with university degrees show the highest subscription rate (17.8%) — 1.6x overall average
- **Low-value segment:** blue-collar workers consistently show the lowest subscription rates (<7.5%)
- **Loyalty effect:** previous subscribers have a 65% re-subscription rate — 6x overall average
- **Channel preference:** cellular outperforms telephone (14.3% vs ~5%)
- **Seasonal pattern:** March and October show highest subscription rates; May shows lowest
- **Data leakage:** `duration` was retained for EDA but excluded before modelling

---

## Part 2 — Modelling & Business Analysis

**Notebook:** `36103_25P_AT3_Code_final_github.ipynb`

### Feature Engineering
Three binary features were created based on EDA findings:
- `young_uni` — aged 20–40 with university or high school degree
- `has_previous_contact` — whether the client was contacted before this campaign
- `low_campaign` — whether the number of contacts in this campaign was 2 or fewer

### Preprocessing
- Stratified train / validation / test split (35% / 35% / 30%)
- One-Hot Encoding for categorical features (fit on train only)
- StandardScaler for numerical features (fit on train only)
- SMOTE applied on training set (sampling strategy = 0.35)
- `duration` and `age` dropped to prevent data leakage and multicollinearity

### Models Trained
| Model | Class 1 F1 | ROC-AUC |
|---|---|---|
| Dummy Classifier (baseline) | — | — |
| Logistic Regression (GridSearchCV) | 0.44 | 0.78 |
| LightGBM (GridSearchCV) | 0.48 | 0.79 |
| LightGBM + SMOTE | lower than baseline | — |

**Selected model:** LightGBM (baseline, without SMOTE)  
**Recommended threshold:** 0.4 — captures 60% of potential subscribers (Recall = 0.603) while maintaining acceptable precision (0.377)

### Business Analysis
- **Top 10% analysis:** model's top 10% predicted customers achieve a 49.93% actual subscription rate — 4.5x overall average
- **High-value segment** (university degree + no default + previous success): subscription rate 66.6%, 16.9x more likely to subscribe (OR = 16.9, p < 0.001)
- **Low-value segment** (blue-collar + unknown default): subscription rate 4.8%, significantly below average (Z-test p < 0.001)
- **Simulation:** cellular contact in March achieves the highest subscription rate (80%) for the high-value segment

### Key Recommendations
- Prioritise customers aged 20–30 with university degrees and previous subscribers
- Contact via cellular; limit to 1–2 contacts per customer
- Schedule campaigns in March or October; avoid May
- Deprioritise blue-collar workers with unknown credit status

---

## Part 3 — API Deployment

The trained model is deployed as a REST API using FastAPI and Docker, hosted on Render.

### How It Works
Send a POST request to `/predict` with customer data, and the API returns the subscription probability and prediction.

**Endpoint:** `POST /predict`

**Example Request:**
```json
{
  "age": 35,
  "job": "admin.",
  "marital": "single",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "mon",
  "campaign": 1,
  "pdays": -1.0,
  "previous": 0,
  "poutcome": "nonexistent",
  "emp_var_rate": -1.8,
  "cons_price_idx": 92.893,
  "cons_conf_idx": -46.2,
  "euribor3m": 1.299,
  "nr_employed": 5099.1,
  "age_group": "30-60"
}
```

**Example Response:**
```json
{
  "subscription_probability": 0.3114,
  "prediction": 0
}
```

### Run Locally with Docker
```bash
docker build -t tele-prediction .
docker run -p 8000:8000 tele-prediction
```
Then open `http://127.0.0.1:8000/docs`

---

## Tech Stack

- **Modelling:** Python, pandas, numpy, scikit-learn, LightGBM, imbalanced-learn
- **Analysis:** matplotlib, seaborn, statsmodels, scipy
- **Deployment:** FastAPI, Docker, Render
