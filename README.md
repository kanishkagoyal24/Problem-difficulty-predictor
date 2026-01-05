# 🧠 **ACM Problem Difficulty Predictor**

A full-stack Machine Learning project to predict the difficulty level and numerical complexity score of algorithmic programming problems using natural language text and engineered features. Powered by supervised learning and deployed with an intuitive web interface.

---

## 📌 **Project Overview**

The ACM Problem Difficulty Predictor is an end-to-end machine learning system that estimates how hard a competitive programming problem is based on its text description. Given a problem statement, input/output specification, and examples, the system predicts:

* ✨ A categorical difficulty class: **Easy / Medium / Hard**
* 📊 A numeric difficulty score (**0–10 scale**)

This helps learners, educators, and platforms automate problem labeling or create adaptive learning pipelines.

---

## 📂 **Dataset Used**

We use the **TaskComplexity dataset**, which contains **4,112 real programming tasks** extracted from various coding sites via web scraping. Each entry includes:

* Title of the problem
* Problem Statement
* Input/Output Description
* Complexity Category (Easy / Medium / Hard)
* Complexity Score (numeric)

**Dataset link:** [https://github.com/AREEG94FAHAD/TaskComplexityEval-24](https://github.com/AREEG94FAHAD/TaskComplexityEval-24)

---

## 🧠 **Approach and Models Used**

### 🛠️ **Feature Engineering**

* Text cleaning and preprocessing
* TF-IDF vectorization (unigrams + bigrams)
* Engineered numeric features:

  * Text length
  * Word count
  * Constraint density
  * Count of numeric tokens

---

### 🤖 **Models**

We trained and compared multiple models for both classification and regression tasks.

#### 🔹 **Classification Models (Difficulty Class: Easy / Medium / Hard)**

* Logistic Regression
* Linear Support Vector Classifier (LinearSVC)
* Multinomial Naive Bayes
* Random Forest Classifier

**✅ Best Classification Model:** **Random Forest Classifier**
Chosen because it achieved the highest validation accuracy and handled nonlinear interactions between features better than linear models.

---

#### 🔹 **Regression Models (Difficulty Score: 0–10 scale)**

* Linear Regression
* Ridge Regression
* Lasso Regression
* XGBoost Regressor
* LightGBM Regressor

**✅ Best Regression Model:** **XGBoost Regressor**
Chosen due to its superior performance in terms of RMSE and MAE, and its ability to generalize better on high-dimensional sparse text features.

---

## 📊 **Evaluation Metrics**

We evaluated multiple models for both classification and regression tasks using standard metrics. This section first compares all models and then summarizes the best-performing ones.

### 🔹 **Classification Performance (Easy / Medium / Hard)**

| Model                    | Accuracy | Precision | Recall | F1-score |
| ------------------------ | -------- | --------- | ------ | -------- |
| Logistic Regression      | ~0.45    | ~0.44     | ~0.45  | ~0.44    |
| Linear SVC               | ~0.47    | ~0.46     | ~0.47  | ~0.46    |
| Multinomial Naive Bayes  | ~0.42    | ~0.41     | ~0.42  | ~0.41    |
| Random Forest Classifier | ~0.52    | ~0.51     | ~0.52  | ~0.51    |

**Observation:** Random Forest outperforms linear and probabilistic models, indicating that nonlinear feature interactions are important for difficulty classification.

---

### 🔹 **Regression Performance (Difficulty Score Prediction)**

| Model              | MAE ↓ | RMSE ↓ |
| ------------------ | ----- | ------ |
| Linear Regression  | ~2.30 | ~2.90  |
| Ridge Regression   | ~2.10 | ~2.60  |
| Lasso Regression   | ~2.05 | ~2.55  |
| XGBoost Regressor  | ~1.70 | ~2.03  |
| LightGBM Regressor | ~1.80 | ~2.15  |

**Observation:** Gradient boosting models significantly outperform linear models, with XGBoost achieving the lowest error.

---

### 🏆 **Best Models Summary**

| Task           | Best Model               | Metric          | Reason                                                  |
| -------------- | ------------------------ | --------------- | ------------------------------------------------------- |
| Classification | Random Forest Classifier | Accuracy ≈ 0.52 | Best performance on imbalanced, nonlinear feature space |
| Regression     | XGBoost Regressor        | RMSE ≈ 2.03     | Lowest prediction error and best generalization         |

---

## **Final Selection**

* **Final Classification Model:** Random Forest Classifier
* **Final Regression Model:** XGBoost Regressor

These models were chosen for deployment in the web application due to their superior performance on validation data and robustness to high-dimensional sparse features.

---

## ⚡ **Steps to Run the Project Locally**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/Problem-difficulty-predictor
cd Problem-difficulty-predictor
```

### 2️⃣ Set up environment

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3️⃣ Run the web interface

```bash
streamlit run app.py
```

Then open the link shown (usually `http://localhost:8501`).

---

## 🌐 **Web Interface — How It Works**

The Streamlit web app enables users to paste:

* ✏️ Problem Description
* 📥 Input Description
* 📤 Output Description

When the user clicks **“Analyze Difficulty”**, the app:

* Cleans and preprocesses the text
* Generates TF-IDF + engineered features
* Runs the classification and regression models

Shows:

* ✔️ Predicted difficulty label (easy/medium/hard)
* ✔️ A numeric difficulty score

The layout is designed for clarity and real-time feedback.

---

## 🎥 **Demo Video**

📹 **Watch the project demo (2–3 min):**
👉 [https://your-demo-video-link.com](https://your-demo-video-link.com)

---

## 👤 **Author**

**Kanishka Goyal**
📍 Student
🎓 IIT Roorkee
💡 Interests: ML, Data Science, AI Systems

🔗 GitHub: [https://github.com/your-username](https://github.com/your-username)
🔗 LinkedIn: [https://linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)

---

If you want, I can help you add badges (Python version, Streamlit, license, etc.) or center the header — but this version is already clean, readable, and professional for GitHub 😄
