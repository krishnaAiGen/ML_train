#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:40:51 2024

@author: krishnayadav
"""

from langchain_community.llms import Ollama
import pandas as pd
import os

class Summarization:
    def __init__(self, model):
        self.model = model
    
    def summarize_data(self, post_df, coin):
        columns = ['protocol', 'post_id', 'timestamp', 'title', 'description']
        filtered_descriptions = pd.DataFrame(columns= columns)
        for index, row in post_df.iterrows():
            timestamp = row['timestamp']
            protocol = row['protocol']
            title = row['title']
            description = row['description']
            post_id = row['post_id']

        
            llm = Ollama(model=self.model, temperature=0.3)
            if len(description.split(' ')) >= 100:
                prompt = f"summarize the sentiment of following text. remove any integer value or web page link and any other noise and limit the output in between 30-60 words: {description}"
            
            if len(description.split(' ')) < 100:
                prompt = f"summarize the sentiment of following text. remove any integer value or web page link and any other noise and limit the output in between 10-20 words: {description}"

            
            output = llm.invoke(prompt)
            print("\n\n", coin)
            # print("\n PROMPT: ", prompt)
            # print("\n OUTPUT: ", output)
            RED = '\033[91m'
            GREEN = '\033[92m'
            RESET = '\033[0m'
            
            # Printing prompt in red and output in green on separate lines
            print(f"\n{RED}PROMPT:{RESET} {RED}{prompt}{RESET}")
            print(f"\n{GREEN}OUTPUT:{RESET} {GREEN}{output}{RESET}")

            new_row = {
                'timestamp': timestamp, 
                'protocol' : protocol,
                'post_id' : post_id,
                'title' : title,
                'description': output
                }
            
            filtered_descriptions = pd.concat([filtered_descriptions, pd.DataFrame([new_row])], ignore_index=True)
        
        return filtered_descriptions
    
    def get_summary(self, post_df_dict):
        for coin, post_df in post_df_dict.items():
            filtered_descriptions = self.summarize_data(post_df, coin)

            filtered_descriptions.to_csv("f/Users/krishnayadav/Documents/aiTradingBot/data/post_proposal_summarized/{coin}.csv")
            

def read_files(post_directory):
    
    post_df_dict = {}
    for post in os.listdir(post_directory):
        if post == '.DS_Store':
            continue
        
        post_df = pd.read_csv(post_directory + '/' + post)
        post_df_dict[post.split('.')[0].split('_')[0]] = post_df
        
    return post_df_dict
    
if __name__ == "__main__":
    summarize = Summarization("mistral")
    post_directory = '/Users/krishnayadav/Documents/aiTradingBot/data/post_proposal_cleaned/'
    post_df_dict = read_files(post_directory)
    summarize.get_summary(post_df_dict)
    
    
    



