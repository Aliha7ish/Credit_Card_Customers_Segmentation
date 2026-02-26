# 📊 Credit Card Customers Segmentation  
### Production-Grade Customer Segmentation & Portfolio Strategy

---

## 🎯 Executive Overview

This project applies advanced unsupervised learning techniques to segment credit card customers based on behavioral and financial patterns.

The objective is not only clustering — but delivering:

- Revenue growth strategy
- Credit risk containment
- Portfolio stability
- Long-term customer value optimization

The final output is a deployable segmentation pipeline with clear commercial interpretation and actionable business strategy.

---

# 🧠 Business Problem

Financial institutions manage diverse customer portfolios with varying:

- Spending behavior
- Repayment discipline
- Credit exposure
- Liquidity dependence

Without segmentation, strategy becomes generic and inefficient.

This project builds a structured segmentation framework to:

- Identify high-value growth segments  
- Detect high-risk revolvers  
- Separate liquidity-driven borrowers  
- Optimize credit and marketing allocation  

---

# 📊 Dataset Overview

- ~9,000 credit card customers  
- 6 months of behavioral history  
- 17 financial and transactional features  

### Key Variables

- **Balance**
- **Purchases**
- **Cash Advances**
- **Credit Limit**
- **Minimum Payments**
- **Purchase Frequency**
- **Installment Purchases**
- **Full Payment Ratio**

Dataset file: `dataset/raw/CC GENERAL.csv`

---

# ⚙️ Methodology

## 1️⃣ Data Preprocessing Pipeline

To ensure structural stability and noise reduction:

```
Raw Data
   ↓
Log2 Transformation (skewness correction)
   ↓
Feature Scaling
   ↓
PCA (6 Components – 93% Variance Explained)
   ↓
HDBSCAN Clustering
```

### Why This Pipeline?

- Log transformation reduces heavy financial skew
- Scaling standardizes behavioral magnitudes
- PCA preserves structure while reducing dimensionality
- HDBSCAN provides density-aware segmentation with noise handling

---

# 🔬 Model Selection Strategy

We evaluated:

- DBSCAN
- HDBSCAN
- OPTICS

Across:

- Original scaled dataset (17 features)
- PCA-reduced dataset

---

## 📉 PCA Justification

<p align="center">
  <img src="./assets/PCA&apos;s Across Different Scaling Techniques.png" width="100%">
</p>

The first **6 principal components capture ~93% of total variance**, meaning:

- Dimensionality reduced from 17 → 6
- Minimal information loss
- Improved cluster compactness
- Stronger separation
- Reduced outlier inflation

PCA significantly improved interpretability and stability.

---

## 🧪 Model Comparison Summary

### 🔹 OPTICS
- Generated 50+ clusters
- High overlap
- Over-fragmented segmentation
- Poor business usability

### 🔹 DBSCAN
- More reasonable cluster count
- Some overlapping clusters (7, 8, 9)
- t-SNE suggests potential merging

### 🔹 HDBSCAN ✅ (Selected Model)
- 9 clusters (including outliers)
- Clear separation
- Limited overlap
- Stable density-based structure
- Commercially interpretable segments

<p align="center">
  <img src="./assets/tsne optics.png" width="40%">
</p>
<p align="center">
  <img src="./assets/tsne dbscan.png" width="40%">
</p>
<p align="center">
  <img src="./assets/tsne hdbscan.png" width="40%">
</p>

---

## 📈 Model Comparison & Visual Validation

| Model        | Observations | Cluster Visualization |
|--------------|-------------|-----------------------|
| **OPTICS**   | Over-fragmented clusters with significant overlap. Generated 50+ micro-clusters, reducing interpretability. | <img src="./assets/optics%20clusters.png" width="80%"> |
| **DBSCAN**   | More reasonable segmentation than OPTICS, but some clusters (7, 8, 9) appear close and potentially mergeable. | <img src="./assets/dbscan%20clusters.png" width="80%"> |
| **HDBSCAN ✅** | Best structural separation, limited overlap, and commercially interpretable segmentation (~8 meaningful clusters). | <img src="./assets/hdbscan%20clusters.png" width="80%"> |

---

# 🏁 Final Model Decision

> **HDBSCAN trained on PCA-reduced data (6 components, 93% variance).**

This model offers:

- Balanced segmentation (~8 meaningful clusters)
- Robust outlier handling
- High structural clarity
- Production scalability
- Strategic interpretability

---

# 📌 Customer Segmentation & Strategic Actions

The segmentation produces 6 core behavioral archetypes.

---

## 🟢 1. Whale Power Users (Cluster 5)

### Profile
- Highest purchases (~2,400)
- Highest purchase frequency
- Highest installment usage
- Highest credit limits (~5,800)

### Behavior
Active, financially sophisticated, high engagement.

### Strategy
- Premium rewards & concierge services
- Selective credit limit increases
- Luxury/travel partnerships
- BNPL expansion

**Objective:** Maximize lifetime value & wallet share.

---

## 🔴 2. High-Risk Revolvers (Clusters 6 & 7)

### Profile
- High balances (~2,700)
- High minimum payments (~1,500)
- Near-zero full payment rates
- High cash advance activity

### Risk
Repayment strain & potential delinquency exposure.

### Strategy
- Restrict aggressive credit expansion
- Offer structured consolidation plans
- Early intervention & financial wellness outreach
- Controlled risk containment

**Objective:** Stabilize risk while preserving yield.

---

## 🟠 3. Cash-Advance Borrowers (Clusters 2 & 3)

### Profile
- Highest cash advance usage (~2,000)
- Low retail purchases
- Liquidity-driven behavior

### Interpretation
Card used as short-term credit instrument.

### Strategy
- Migrate to structured personal loans
- Reduce high-cost revolving dependency
- Introduce liquidity planning tools

**Objective:** Convert unstable revolving usage into structured lending.

---

## 🔵 4. Disciplined Transactors (Clusters 0 & 4)

### Profile
- Moderate spending
- Highest full payment rate (~0.32)
- Low balances (~800)

### Risk
Very low credit risk.

### Strategy
- Cashback optimization
- Merchant partnerships
- Increase transaction volume
- Encourage recurring payments

**Objective:** Maximize interchange revenue with minimal risk exposure.

---

## 🟣 5. Dormant / Inactive (Cluster 1)

### Profile
- Near-zero balances
- Minimal activity
- Short tenure

### Strategy
- Controlled reactivation campaigns
- First-purchase incentives
- Engagement surveys
- Cost-controlled testing

**Objective:** Selective revenue activation.

---

## 🟡 6. Big-Ticket Outliers (Cluster -1)

### Profile
- High one-off purchases
- High payments
- Low frequency
- Large transaction exposure

### Strategy
- Premium retail partnerships
- Seamless large transaction approvals
- Enhanced fraud monitoring
- Relationship management for high-limit clients

**Objective:** Capture high-margin transactions while managing exposure.

---

# 📊 Portfolio-Level Strategic Summary

The portfolio divides into:

- Growth drivers → Cluster 5
- Low-risk revenue engines → Clusters 0 & 4
- Yield-heavy but risky revolvers → Clusters 6 & 7
- Liquidity-dependent borrowers → Clusters 2 & 3
- Reactivation candidates → Cluster 1
- High-exposure outliers → Cluster -1

---

# 🛠 Tech Stack

- Python
- NumPy
- pandas
- Scikit-learn
- HDBSCAN
- Matplotlib
- Plotly
- Seaborn
- PCA
- t-SNE (visual validation)

---

# 🚀 Production Value

This project demonstrates:

- Advanced unsupervised learning
- Dimensionality reduction reasoning
- Model comparison methodology
- Risk-aware business translation
- Strategic portfolio thinking
- Production-ready ML pipeline design

---

# 📌 Author

Ali Hashish  
Machine Learning Engineer  

---
