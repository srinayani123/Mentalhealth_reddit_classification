# 🧠 Mental Health Reddit Post Classifier and Triage Score Estimator

This project builds and evaluates NLP models to classify mental health conditions and assign a triage severity score to Reddit posts, enabling early identification and prioritization of at-risk users in online communities.

---

## 📌 Problem Statement

Online mental health communities often receive thousands of posts daily, many of which contain subtle signs of emotional distress or suicidal ideation. Manual moderation is time-consuming and inconsistent. The goal is to build a system that can automatically classify posts into conditions (e.g., depression, anxiety) and estimate the urgency (triage score) to assist in early intervention.

---

## ✅ Proposed Solution

I fine-tuned multiple transformer models on annotated Reddit data to classify posts into five mental health categories and regress a continuous triage score based on linguistic patterns of risk. This dual modeling approach allows both categorical tagging and prioritization of cases needing attention.

---

## ⚙️ Methodology

1. **Data Preprocessing**  
   - Combined `title` and `text` fields.
   - Handled missing values.
   - Mapped target labels and created `triage_score` using regex-based heuristics.

2. **Model Selection**  
   - **Classification**:  
     - `bert-base-uncased`  
     - `distilbert-base-uncased`  
     - `roberta-base`  
     - `Sharpaxis/Mental-Health-RoBERTa` (domain-specific)  
   - **Regression**:  
     - Same models adapted for single-output regression.

3. **Tokenization & Training**  
   - Used respective model tokenizers with truncation/padding.
   - Applied stratified train/test split.
   - Fine-tuned models using `Trainer` with evaluation metrics: accuracy (classification), MSE (regression).

4. **Triage Scoring**  
   - Regex patterns categorized language into low/mild/moderate/high-risk.
   - High-risk samples were upweighted to ensure model learns severity scale.

---

## 📊 Evaluation Metrics

- **Classification**:
  - Accuracy, Precision, Recall, F1 Score (weighted + per class)
- **Regression**:
  - Mean Squared Error (MSE), qualitative analysis of predicted vs expected scores.

---

## 📈 Results (Best Model)

- **Classifier**: `Sharpaxis/Mental-Health-RoBERTa`
  ![image](https://github.com/user-attachments/assets/80b61743-127e-4904-adb9-8ba52bac7ead)

  - Strong performance on subtle classes like bipolar/personality disorder.
  
- **Regressor**: `distilbert-base-uncased` (triage scoring)
  - Captured risk ranges from 0.02 to 1.00 effectively.
  - Outperformed other models in semantic generalization.

---

## ✅ Conclusion & Model Justification

The `Mental-Health-RoBERTa` model achieved the best balance of accuracy and class-wise recall, making it suitable for classification in sensitive clinical scenarios. For regression, `DistilBERT` provided smoother generalization and faster training without sacrificing much accuracy. Domain-specific pretraining proved highly beneficial in understanding nuanced mental health expressions.

---

## 💻 Run Instructions

```bash
git clone https://github.com/srinayani123/Mentalheakth_reddit_classification
cd mental_health
pip install -r requirements.txt
