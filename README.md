# 🧠 Mental Health Reddit Post Classifier and Triage Score Estimator

This project builds and evaluates NLP models to classify mental health conditions and assign a triage severity score to Reddit posts, enabling early identification and prioritization of at-risk users in online communities.

---

## 📌 Problem Statement

Online mental health communities often receive thousands of posts daily, many of which contain subtle signs of emotional distress or suicidal ideation. Manual moderation is time-consuming and inconsistent. The goal is to build a system that can automatically classify posts into conditions (e.g., depression, anxiety) and estimate the urgency (triage score) to assist in early intervention.

---

## ✅ Proposed Solution

I fine-tuned multiple transformer models on annotated Reddit data to classify posts into five mental health categories and regress a continuous triage score based on linguistic patterns of risk. This dual modeling approach allows both categorical tagging and prioritization of cases needing attention.

---
## 📂 Data Description

The dataset consists of Reddit posts collected from various mental health-related subreddits. Each record contains:

- `title`: Post title
- `text`: Body of the post
- `target`: Integer label (0–4), representing one of five mental health conditions:
  - **0** – Depression  
  - **1** – Anxiety  
  - **2** – Bipolar Disorder  
  - **3** – Personality Disorder  
  - **4** – Stress

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

  ![image](https://github.com/user-attachments/assets/c684831e-9b12-48da-a35a-4cd3d5af46e1)

  - Captured risk ranges from 0.02 to 1.00 effectively.
  - Outperformed other models in semantic generalization.

---

## Outputs- 

- **Classifier** :
  ![image](https://github.com/user-attachments/assets/632fa594-589b-4652-9345-7aecb106ce7b)
- **Regressor** :
- ![image](https://github.com/user-attachments/assets/6e686fb3-af23-43ad-94f8-9da413d92bad)

----
## ✅ Conclusion & Model Justification

The `Mental-Health-RoBERTa` model achieved the best balance of accuracy and class-wise recall, making it suitable for classification in sensitive clinical scenarios. For regression, `DistilBERT` provided smoother generalization and faster training without sacrificing much accuracy. Domain-specific pretraining proved highly beneficial in understanding nuanced mental health expressions.

---

## 💻 Run Instructions

1. Clone the repository:
   
```bash
git clone https://github.com/srinayani123/Mentalheakth_reddit_classification
pip install -r requirements.txt

2. Then, open the colab
Navigate to model_finetuning or data preprocessing/ and open any .ipynb via Colab.

3. Upload data:
Manually upload data when prompted while executing the code.

4. Run each notebook step by step:
Each notebook is self-contained and runnable from start to end.


