# Sentiment Analysis with Machine Learning and Neural Network

This GitHub repository contains code for performing sentiment analysis on text data using machine learning and a neural network. The analysis includes various steps such as data preprocessing, feature engineering, model selection, and evaluation. Below, I'll explain each section of the code and provide an overview of the project.

## Project Overview
Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) task that involves determining the sentiment or emotion expressed in a piece of text, such as a sentence or a document. In this project, we perform sentiment analysis on a dataset with three sentiment classes: 'Positive', 'Negative', and 'Neutral'. We use both traditional machine learning models (Random Forest, Logistic Regression, and Naive Bayes) and a neural network to classify text data into these sentiment classes.

## Project Structure
The project is organized into several main components:
Dataset Extraction: Load and explore the dataset to understand its structure and characteristics.
Text Preprocessing and Feature Engineering: Preprocess text data by removing special characters, lemmatizing words, and extracting word vectors using pre-trained Word2Vec embeddings.

### Model Selection: 
Train and evaluate machine learning models, including Random Forest, Logistic Regression, and Naive Bayes, using GridSearchCV for hyperparameter tuning.

### Model Evaluation: 
Visualize and compare the accuracy of different models using bar plots.

### Best Model Selection: 
Determine the best-performing model based on accuracy and save it for later use.

### Neural Network: 
Implement a neural network model for sentiment analysis using TensorFlow and Keras. Train and evaluate the model's performance.

### Model Persistence: 
Save the best machine learning model and the neural network model to disk for future use.

### Usage:
Run the Python scripts provided in each section to execute the corresponding tasks.
For the neural network model, you can use the sentiment_analysis_model.h5 file to make predictions on custom input..

## Acknowledgments
This project uses pre-trained Word2Vec embeddings from the "word2vec-google-news-300" model.
Please feel free to contribute to this project or use it as a reference for your own sentiment analysis tasks.
