#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:22:56 2024

@author: krishnayadav
"""

import pandas as pd

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain_community.llms import Ollama
import pandas as pd
import os

class Summarization:
    def __init__(self, model):
        self.model = model
    
    def summarize_text(self, description):
        llm = Ollama(model=self.model, temperature=0.3)
        if len(description.split(' ')) >= 100:
            prompt = f"summarize the sentiment of following text. remove any integer value or web page link and any other noise and limit the output in between 30-60 words: {description}"
        
        if len(description.split(' ')) < 100:
            prompt = f"summarize the sentiment of following text. remove any integer value or web page link and any other noise and limit the output in between 10-20 words: {description}"

        
        output = llm.invoke(prompt)
        
        return output

class FinBERTSentiment:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")    
        self.model.to(self.device)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)        
        inputs = {key: value.to(self.device) for key, value in inputs.items()}   
        
        with torch.no_grad():
            outputs = self.model(**inputs)        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]  # Move back to CPU for numpy processing
        
        # Get the predicted class (0=negative, 1=neutral, 2=positive)
        predicted_class = np.argmax(probabilities)
        labels = ['negative', 'neutral', 'positive']
        sentiment = labels[predicted_class]
        
        return sentiment, max(probabilities)


import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

class CryptoSentimentAnalyzer:
    def __init__(self, model_name="kk08/CryptoBERT"):
        # Check if CUDA is available and set device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model on the specified device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name).to(self.device)

        # Initialize sentiment analysis pipeline with the correct device index
        device_index = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=device_index)

        # Dictionary to map model labels to sentiment labels
        self.sent_dict = {
            "LABEL_0": "negative",
            "LABEL_1": "positive"
        }

    def analyze_sentiment(self, text):
        try:
            result = self.classifier(text)[0]
            label = result['label']
            prob = result['score']
            return self.sent_dict.get(label, "unknown sentiment"), prob
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return None

def predict_final_sentiment(sentiment, sentimnet_score, sentiment_crypto, crypto_score):
    if sentiment != sentiment_crypto:
        return sentiment, sentimnet_score
    
    else:
        sentimnet_score = (sentimnet_score + crypto_score) / 2
        return sentiment, sentimnet_score

def predict_sentiment_both():
    finbert = FinBERTSentiment()
    crypto = CryptoSentimentAnalyzer()
    summarize_obj = Summarization("mistral")
    
    sentiment_score_list = []
    sentiment_boolean_list = []
    
    cleaned_data = pd.read_csv("cleaned_proposal26oct.csv")
    cleaned_data = cleaned_data[:10]
    for index, row in cleaned_data.iterrows():
        print(index)
        try:
            text = row['proposal']
            sentiment_type = row['proposal_type']
            text = summarize_obj.summarize_text(text)
    
            sentiment1, sentimnet_score1 = finbert.predict(text)
            sentiment2 , sentimnet_score2 = crypto.analyze_sentiment(text)
            sentiment, sentimnet_score = predict_final_sentiment(sentiment1, sentimnet_score1, sentiment2 , sentimnet_score2)
            
            #positive condition
            if sentiment == 'positive' and sentiment_type == 'bullish' and sentimnet_score > 0.8:
                sentiment_score_list.append(sentimnet_score)
                sentiment_boolean_list.append(True)
            
            #negative condition
            elif sentiment == 'negative' and sentiment_type == 'bearish' and sentimnet_score > 0.8:
                sentiment_score_list.append(sentimnet_score)
                sentiment_boolean_list.append(True)
            
            #neutral condition
            elif 0.5 < sentimnet_score < 0.7:
                sentiment_score_list.append(sentimnet_score)
                sentiment_boolean_list.append(True)
            
            #False condition
            else:
                sentiment_score_list.append(sentimnet_score)
                sentiment_boolean_list.append(False)
        
        except:
            continue
    
    cleaned_data['sentiment_score'] = sentiment_score_list
    cleaned_data['sent_boolean'] = sentiment_boolean_list
    
    cleaned_data.to_csv('cleaned_proposal28oct.csv')

if __name__ == "__main__":
    predict_sentiment_both()

    

            
            

        
            
        
        
        
    
    
    


