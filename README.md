# Fake News Prediction with Machine Learning

## Description
This project aims to predict fake news using machine learning techniques.Fake news detection is a crucial task in today's digital age to combat misinformation and maintain the integrity of information dissemination channels.

## Datasheet 
The dataset used for training and testing the models can be found [here](https://www.kaggle.com/c/fake-news/data?select=train.csv).

## Techniques Used
- **Logistic Regression**: Logistic Regression from the [scikit-learn](https://scikit-learn.org/stable/) library is utilized as one of the machine learning algorithms for binary classification
- **Natural Language Toolkit (NLTK)**: [NLTK](https://www.nltk.org/) is utilized for natural language processing tasks including tokenization, stopword removal, and stemming.
- **Stemming Process**: Stemming, a part of the text preprocessing pipeline, involves reducing words to their root or base form using NLTK's stemming algorithms such as the Porter Stemmer and Lancaster Stemmer. This helps in normalizing words and reducing vocabulary size, thereby improving the performance of machine learning models by reducing noise in the data.

## Natural Language Toolkit (NLTK) and Stemming Process
[NLTK](https://www.nltk.org/) (Natural Language Toolkit) is a powerful Python library for natural language processing (NLP) tasks. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning. In this project, NLTK is utilized for text preprocessing tasks such as tokenization, stopword removal, and stemming.

Stemming is the process of reducing words to their root or base form. This is beneficial in text analysis tasks as it helps to normalize words and reduce the vocabulary size, which can improve the performance of machine learning models by reducing noise in the data. NLTK provides various stemming algorithms such as the Porter Stemmer and Lancaster Stemmer, which can be employed based on the requirements of the project.

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/Himu336/Fake-News-Prediction_ML.git

## Contributions
Contributions to improve the code, add features, or fix issues are welcome. Feel free to fork the repository and submit pull requests.

