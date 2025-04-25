## Overview

This repository contains a machine learning-based mental health risk assessment system that analyzes text for indicators of depression and self-harm. The system combines advanced NLP techniques with feature engineering to achieve high accuracy in identifying concerning patterns.

### Key Features

- **Text Analysis**: Advanced NLP pipeline using sentence embeddings and linguistic feature extraction
- **Risk Assessment**: Machine learning model trained to identify potential mental health concerns
- **Temporal Tracking**: Monitoring changes in risk levels over time
- **User-Friendly Interface**: Clean, accessible web application for easy interaction
- **Resource Integration**: Connection to mental health support resources

## Model Performance

Our optimized Histogram Gradient Boosting Classifier significantly outperforms baseline approaches:

|
 Model                    
|
 Accuracy 
|
 Precision 
|
 Recall 
|
 F1 Score 
|
|
--------------------------
|
----------
|
-----------
|
--------
|
----------
|
|
 Logistic Regression      
|
 69.2%    
|
 0.694     
|
 0.692  
|
 0.691    
|
|
 Hist. Gradient Boosting  
|
 87.5%    
|
 0.889     
|
 0.874  
|
 0.873    
|

* ROC AUC: 0.932 (compared to 0.758 for Logistic Regression)
* Average Precision: 0.949 (compared to 0.785 for Logistic Regression)

## Technology Stack

- **Backend**: Flask web framework
- **Machine Learning**: Scikit-learn, Sentence Transformers
- **NLP**: NLTK, TextBlob
- **Frontend**: HTML/CSS, Bootstrap, Chart.js
- **Data Processing**: Pandas, NumPy


 
