# 360Churn — Decision-Aware Customer Retention Analytics

## 📌 Overview
**360Churn** is a research-oriented, decision-aware churn analytics system that extends
traditional churn prediction by integrating **customer lifetime value (CLV)** and
**retention cost** into a unified decision framework.

Unlike probability-only churn models, this system supports
**budget-constrained retention decisions**, enabling organizations to allocate limited
retention resources more effectively.

This project was developed as an **MS Data Science Capstone** and is suitable for
both **academic research** and **industry deployment**.

---

## 🚀 Live Demo
👉 **https://360churn.streamlit.app**

*(Interactive Streamlit dashboard with portfolio analytics and budget simulation)*

---

## ✨ Key Features
- Machine learning–based churn prediction
- Decision-aware customer prioritization
- Customer lifetime value (CLV) estimation
- Budget-constrained retention optimization
- ROI and budget sensitivity analysis
- Power BI–style interactive dashboards
- End-to-end Streamlit web application

---

## 🧠 Methodology
1. Preprocessed the Telco Customer Churn dataset
2. Trained a Random Forest churn prediction model
3. Estimated customer lifetime value using tenure and billing data
4. Designed a **decision priority score** combining:
   - churn probability
   - expected revenue loss
   - retention cost
5. Implemented a **budget-constrained optimization strategy**
6. Compared decision-aware retention against probability-only baselines

---

## 📊 Data
- **Dataset:** Telco Customer Churn (IBM Sample Data / Kaggle)
- **Currency Assumption:**  
  Monetary values are treated as **USD for interpretability**, consistent with the
  dataset’s origin.
- Raw dataset is not included due to licensing restrictions.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Plotly, Matplotlib, Seaborn  
- **Web App:** Streamlit  
- **Deployment:** Streamlit Community Cloud  

---

## 🔬 Research Contribution
This project demonstrates that **decision-aware retention strategies** can significantly
outperform probability-only churn models under realistic budget constraints.

Key contributions include:
- Bridging predictive modeling with managerial decision-making
- Explicit modeling of economic value and retention cost
- Interactive simulation of budget allocation policies

---

## ▶️ Run Locally
```bash
streamlit run app.py
