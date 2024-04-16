# Sentiment Analysis of Indian Political Tweets using LSTM

This project was developed as part of the Advanced Machine Learning course requirement, focusing on analyzing the sentiment of Indian political tweets using a Long Short-Term Memory (LSTM) model. The study efficiently incorporates Tweepy for data collection, utilizes a labeled dataset from Kaggle, and applies Natural Language Processing (NLP) techniques along with Global Vector (GloVe) word embeddings for data preprocessing. The project's methodology offers a comprehensive approach to sentiment analysis, categorizing tweets into positive, neutral, or negative sentiments and achieving an impressive model accuracy of 96%.

## Dataset

The dataset for this project was both scraped by Tweepy and sourced from Kaggle, specifically tailored for sentiment analysis of Indian political tweets. It plays a crucial role in training and validating our LSTM model to ensure accurate sentiment classification.

- **Dataset Source**: [Kaggle: Indian Political Tweets Sentiment Analysis](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset)

## Models and Accuracy

The LSTM model was employed for sentiment classification, with the following performance metrics:

- Precision
- Recall
- F1-Score

These metrics indicate the model's strong ability to classify the sentiment of tweets accurately. The performance is summarized as follows:

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.95      | 0.96   | 0.96     | 7276    |
| 1         | 0.95      | 0.97   | 0.96     | 6935    |
| 2         | 0.97      | 0.94   | 0.95     | 7089    |
|           |           |        |          |         |
| Accuracy  |           |        | 0.96     | 21300   |
| Macro Avg | 0.96      | 0.96   | 0.96     | 21300   |
| Weighted Avg | 0.96   | 0.96   | 0.96     | 21300   |

- [**Click here for Source Codes**](https://github.com/invcble/Sentiment-Analysis-of-Indian-Political-Tweets-2023/tree/ec49ca15b794566ff53c79ab2bfa2437bc95431b/Source%20codes)
- [**Click here for Project Report**](https://github.com/invcble/Sentiment-Analysis-of-Indian-Political-Tweets-2023/blob/ec49ca15b794566ff53c79ab2bfa2437bc95431b/Project_Report_7thSEM.pdf)

## Comparative Sentiment Analysis

In addition to the LSTM model's sentiment analysis, a comparative analysis was conducted with the Valence Aware Dictionary and sEntiment Reasoner (VADER), a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.

### Comparative Sentiment Analysis for BJP and INC Tweets

| Party | Model | Positive | Neutral | Negative |
|-------|-------|----------|---------|----------|
| BJP   | LSTM  | 48.8%    | 17.4%   | 33.8%    |
| BJP   | VADER | 49.2%    | 15.7%   | 35.1%    |
| INC   | LSTM  | 49.4%    | 17.3%   | 33.3%    |
| INC   | VADER | 48.5%    | 16.7%   | 34.8%    |

#### Visual Comparison
![Pie Chart of BJP Tweets (Predicted By Our Model) vs (Predicted By VADER)](https://github.com/invcble/Sentiment-Analysis-of-Indian-Political-Tweets-2023/assets/58978137/98cca5de-0b66-4664-8dff-3fda0a91e75b)
![Pie Chart of INC Tweets (Predicted By Our Model) vs (Predicted By VADER)](https://github.com/invcble/Sentiment-Analysis-of-Indian-Political-Tweets-2023/assets/58978137/434575e5-23a7-4d38-9230-70017ac70e9e)

### Interpretation and Insights

By visually comparing the two sets of pie charts for each political party (BJP and INC), we can observe the similarities and differences in sentiment distributions. This comparison helps us assess the alignment between our model's predictions and VADER's predictions for the given political teams. It provides insights into the model's performance relative to an established sentiment analysis tool like VADER. The slight variations in the sentiment distributions also offer an opportunity to explore the nuances captured by our LSTM model versus the heuristic approach employed by VADER.

The close alignment in overall sentiment distribution for both the BJP and INC tweets between the LSTM model and VADER suggests that the LSTM model is quite robust and aligns well with conventional sentiment analysis methods. However, the differences in the neutral and negative categories invite further exploration into the linguistic subtleties and context that may influence the sentiment analysis results.


## Setup and Running the Project

To replicate and run this project, follow these steps:

1. **Collect Tweets**: Use Tweepy to collect Indian political tweets. A guide for setting up Tweepy and collecting tweets is provided in the source codes.
2. **Prepare the Dataset**: Download the labeled dataset from Kaggle and preprocess it using the provided scripts for GloVe embeddings.
3. **Download GloVe**: Download Global Vector dimension file and place it within the folder. This project used glove.6B.50d.txt
4. **Train and Evaluate the LSTM Model**: Follow the instructions in the LSTM model implementation folder to train and evaluate the sentiment analysis model.

## Requirements

This project is designed to be run in a Python environment with support for Jupyter Notebooks or Google Colaboratory. Key dependencies include TensorFlow, Keras, Tweepy, Pandas, NumPy, and Matplotlib.

## Acknowledgments

We extend our gratitude to the academic staff and my peers at Bengal Institute of Technology for their invaluable feedback and support. Special thanks to the Kaggle community for providing a comprehensive dataset for sentiment analysis of Indian political tweets.

