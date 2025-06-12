# 🎥 YouTube Analytics & Chat Assistant  – Chrome Extension
A full-stack AI-powered Chrome Extension that helps **content creators and viewers** understand YouTube videos better through **comment analysis**, **video summarization**, and  **LLM-powered chatbot**. Integrated with **CI/CD pipelines**, **ML experiments tracking**, and **scalable deployment** using AWS.

---

## 📌 Key Highlights

- 🤖 LLM-powered Chatbot with FAISS (VectorDB) & Hybrid Search (Dense + Sparse)
- 📊 Comment Analysis: Sentiments, Trends, Word Cloud, Summary of comments
- 🎬 Video Summarization using T5-small LLM using Hugging Face
- 🌐 Web Article Fetching using Tavily(Tool) + LLM using Prompting
- 🚀 CI/CD with GitHub Actions, MLflow, DVC, Docker, ECR, CodeDeploy, S3, ASG(Auto Scaling Group )
- 🧪 Py_test suite for model loading, performance, and signature testing
- 🔒 AutoScaling + Load Balancer using AWS for production readiness
- 🎯 Chrome Extension built with HTML, CSS, JS

## 🎥 Demo Video

[![Demo Video](https://img.youtube.com/vi/2s2H7y6Zu5E/0.jpg)](https://youtu.be/2s2H7y6Zu5E)

---

## ✨ Key Features
## 🎨 Creator Mode Dashboard

- **Engagement Analytics**: Total comments, Unique comments, Average commet length, Average sentiment score.
- **Pie chart**: Pie chart for sentiment distribution.
- **Sentiment trend graph**: Sentiment trend of comments on monthly basis
- **Word cloud**: Dynamic word clouds from comments
- **Comments Summary**: AI-powered audience feedback summary using RAG
- **Comments**:Top 25 most impactful comments with sentiment scores

## 👥 Audience Mode Features

- **Video Summarization**: T5-small LLM + Prompting using Hugging Face video content summarization
- **Related Content Articles**: A Real-time Article Fetching using Tavily API(tool) + Gemini LLM
- **Intelligent Chatbot**: RAG Powered + Refine Query + LLM + FAISS(vectorDB)

---
## 🏗️ System Architecture


![alt text](<system_architecture.png>)

---
# Backend Workflow Breakdown

## 1. 📥 Data Ingestion-
* As per the sentiments we requires the labelled data with comments attached to the sentiments which is availbe on the kaggle named Reddit dataset

## 2. 🧹 Data Preprocessing-
* Clean text using:
    * lower comment, remove new line, remove stop words, Lemmatize words, 
* Ensured consistency across training and inference pipelines.

## 3. ⚙️ Feature Selection-
* Used **TF-IDF Vectorizer** to transform comments into numerical feature.
    * Range- Tri-gram
    * Max_Features- 10,000

## 4. 🏋️ Model Building-
* Trained **LightGBM Classifier** for sentiment classification

```
              precision    recall  f1-score   support

          -1       0.81      0.78      0.79      1647
           0       0.84      0.97      0.90      2510
           1       0.92      0.83      0.87      3176

    accuracy                           0.87      7333
   macro avg       0.86      0.86      0.86      7333
weighted avg       0.87      0.87      0.86      7333
```
```python
params = {
    "objective": "multiclass",
    "num_class": 3,         # [Positive, Neutral, Negative]
    "metric": "multi_logloss",
    "is_unbalance": True,   
    "class_weight": "balanced",
    "reg_alpha": 0.1,        # L1 regularization
    "reg_lambda": 0.1,       # L2 regularization
    "learning_rate": 0.09,   # Learning rate was most crucial 
    "max_depth": 20,
    "n_estimators": 367,
}
```
## 5. 📊 Model Evaluation
Evaluated its performance on test dataset-
* Calculated and logged Accuracy, Precision, Recall and F2-Score.
* Computed Confusion Matrix.
* Model Signature- Captured model input & output to ensure expected results.

## 6. 📈 Register Model - MLflow
* Using ML-Flow-
    * Track experiment parameters
    * Register best performing model for production
* All logs and artifact pushed to S3-bucket using EC2 instance.

## 7. 📦 Data Versioning- DVC
* Tracked raw and processed datasets.
* Data and pipelines pushed to S3

## 8. 🧪 Testing - PyTest Suite
* Performed tests using Py-Test-
    * To check ML-Flow working or not. 
    * Promote model from **Staging to Production**.
    * Test application's API.
    * To check model loading from registry.
    * Model performance threshold check.
    * Model Signature validation. 

## 9. 🚀 Deployment with CI/CD
* Created Dockerfile for backend app & pushed to ECR
* Build CI/CD pipelines using **GitHub-Actions** for:
    * Model Deployemnt-
        * Install Dependencies
        * Run Pipeline
        * Push DVC-tracked data to remote
        * Py-Test
        * Build Docker Image, Tag, Push to ECR
*  Used **AWS CodeDeploy** with Auto Scaling Groups and Load Balancer for rolling deployments.

## 10. 🔁 Live Backend App (Flask_API)
* Exposed API for frontend Chrome extension:
    * **/predict**: Predict the sentiments of comments.

    * **/generate_chart**: Generate Pie chart of sentiment of comments
    * **/generate_worldcloud**: Create the wordcloud of Comments.
    * **/generate_trend_graph**: Predict the comments with their time duration, which will displayed on graph.
    * **/summarize_comments**: By using LLM( Gemini-1.5-flash) summarize the audience suggenstion or most discussed topics.**[LLM(Gemini-1.5-flash) + Prompt + VectorDB(FAISS)]**
    * **/summarize_transcript**: By using YT-video transcript - created the Chat-Bot using RAG, which will chat usig video reference. **[LLM(Gemini-1.5-flash) + Query-Refiner + Promopt + RAG + VectorDB(FAISS) + ConversationBufferMemory ]**
    * **/summarize_vide**: Using YT-video transcript summarize the video **[T5-small(Hugging Face Pipeline)]**
    * **/fetch_articles**: Using the summary of video fetch the most recent news articles **[LLM + Prompt + Tavily(Tool)]**




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── Deploy             <- Scripts which will help CodeDeploy to deploy application on EC2
    │
    ├── flask_app          <- Back-end of application
    │                        
    │
    ├── notebooks          <- All the experiments run, before creating final model
    |
    ├── scripts            <- All the PY_Test .py files which will test the application before 
    │                         deploy
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── appspec.yaml       <- Help to CodeDeploy to run scripts before and after app deployment.
    |
    ├── src                <- Source code for use in this project.
    │   |
    │   │
    │   ├── data           <- Scripts to download or generate data & preprocess data.
    │   │   ├──  data_ingestion.py
    |   |   └──  data_preprocessing.py 
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling [TF-IDF]
    │   │   └── feature_selection.py
    │   │
    │   ├── models         <- Scripts to train models, evaluate the model register the model and 
    │       │                 then use trained models to make predictions
    │       ├── model_building.py
    │       ├── model_evaluation.py
    │       └── register_model.py
    |
    ├── Docker              <- To cerate dockerized container to push on ECR
    |
    ├── params.yaml         <- Parameters which used in while creating pipeline  
    │
    └── dvc.yaml            <- To automate the work-flow


--------

