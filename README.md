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
Accuracy : 
 Logistic Regression      
 69.2%    
 
 Hist. Gradient Boosting  
 87.5%    

* ROC AUC: 0.932 (compared to 0.758 for Logistic Regression)
* Average Precision: 0.949 (compared to 0.785 for Logistic Regression)

##Dependencies

pip install flask numpy pandas scikit-learn joblib textblob nltk sentence-transformers

#be sure to grab the NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"


##Required Datasets
Depression cleaned

https://drive.google.com/file/d/1E4VtmgL1tH0d2NeKxmgB21YHKqzgZ-Ph/view?usp=sharing

Reddit multi temporal

https://drive.google.com/file/d/1hgxQGr9ORC8mbGmcGnsUgG5_VY5ONxeb/view?usp=sharing


## Required Files

The application requires the following model artifacts:

- `vectorizer_final.joblib`: Text vectorization model
- `lda_final.joblib`: LDA topic model
- `hgb_final.joblib`: Histogram Gradient Boosting classifier

For the enhanced version with additional features:
- `embedder_mpnet.joblib`: Sentence Transformer model
- `feature_names.joblib`: Feature reference data
- `user_entries.csv`: History tracking database

##How to Use it?

After unzipping the code from GitHub,installing all the dependencies, and acquiring the required dataset, cd into the directory and run the mentalhealth.py app by typing python mentalhealthapplication.py

Please contact me at ojj224@lehigh.edu if something does not work!



 
