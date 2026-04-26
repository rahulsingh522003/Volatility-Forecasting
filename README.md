# 📊 Volatility Forecasting and Risk Modeling

## 🚀 Overview

This project builds a quantitative framework for modeling and forecasting financial market volatility using **EWMA** and **GARCH(1,1)**.

It evaluates performance across **25 liquid equities**, multiple horizons (**1, 5, 10 days**), and extends to **risk estimation using Value at Risk (VaR)** with backtesting.

---

## 🎯 Objectives

- Model time-varying volatility in financial returns  
- Compare EWMA and GARCH across multiple assets  
- Evaluate predictions using QLIKE, RMSE, and MAE  
- Perform multi-horizon analysis (1, 5, 10 days)  
- Estimate Value at Risk (VaR)  
- Validate models using violation rates  

---

## 📦 Project Structure

```
volatility-project/
├── data/
├── results/
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── evaluation.py
│   ├── forecast.py
│   ├── tuning.py
│   ├── risk.py
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
└── README.md
```

---

## 🧠 Methodology

### 1. Data Pipeline
- Downloaded historical price data for **25 liquid equities**
- Computed **log returns**
- Performed **chronological train/validation/test split**

---

### 2. Volatility Models

#### 🔹 EWMA
- Recursive volatility model with fixed decay factor  
- Captures short-term volatility dynamics  

#### 🔹 GARCH(1,1)
- Models volatility using:
  - past shocks (α)
  - past variance (β)
- Learns parameters from data  

---

### 3. Realized Volatility

Used as ground truth:

RV_t = sqrt(sum of squared future returns over horizon)

- Computed using **forward returns**
- Evaluated across **1, 5, and 10-day horizons**

---

### 4. Evaluation Metrics

- **MAE** → average absolute error  
- **RMSE** → penalizes large errors  
- **QLIKE** → preferred metric for volatility models  

---

### 5. Multi-Horizon Analysis

- Evaluated performance across:
  - 1-day (short-term)
  - 5-day (weekly)
  - 10-day (medium-term)

---

### 6. EWMA Lambda Tuning

- Tuned λ over validation set using **QLIKE**
- Observed higher persistence (~0.98) improves performance  

---

### 7. Value at Risk (VaR)

VaR = z * volatility

- Implemented **95% VaR**
- Validated using **violation rates**

---

## 📊 Key Results

- GARCH consistently outperformed EWMA across all stocks  
- Performance gap increased with longer horizons  
- QLIKE showed the most significant improvement  
- GARCH produced better-calibrated VaR estimates  

---

## ▶️ Usage

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

---

## 🧠 Key Concepts Covered

- Volatility clustering  
- EWMA vs GARCH modeling  
- QLIKE-based evaluation  
- Multi-horizon forecasting  
- VaR and risk modeling  
- Backtesting via violations  

---

## 🎯 Applications

- Risk management  
- Portfolio allocation  
- Trading strategies  
- Option pricing  

---

## 🔥 Future Improvements

- GJR-GARCH / EGARCH  
- Heavy-tailed distributions (Student-t)  
- Intraday volatility  
- ML-based models  

---

## 👤 Author

Rahul Singh  
IIT BHU Varanasi  
GitHub: https://github.com/rahulsingh522003  

---

## ⭐ Summary

Data → Volatility Modeling → Evaluation → Risk Estimation
