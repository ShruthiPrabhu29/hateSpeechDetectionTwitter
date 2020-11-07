import json
import pdb
import pandas as pd
import csv

def get_data():
    tweets = []
    data=pd.read_csv("HateDataRSN_16k.csv", engine="python")
    print(data.head())
    print(len(data))
    labelList = data["Class"].tolist()
    textList = data["Tweets"].tolist()
    
    for i in range(len(data)):            
            tweets.append({                
                'text': textList[i],
                'label': labelList[i]               
                })
    
    print(len(tweets))
    return tweets

if __name__=="__main__":
    tweets = get_data()
   # pdb.set_trace()