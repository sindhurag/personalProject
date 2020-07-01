'''
Script to scrape webpage containing 1000 top internet searches in 2018

'''
import pandas as pd
import requests
import numpy as np
import re
from bs4 import BeautifulSoup

convId = []
query = []
awaiting = []
id = 0

outfile = "/Users/iamsgqa/Documents/Automation/Experiments/top1000Searches.xlsx"

url = 'https://www.mondovo.com/keywords/most-asked-questions-on-google'


def checkspecialCharc(dfx):

    id =1
    print(dfx.shape)
    print(dfx.head())
    for s in df['Question']:
        s = ''.join(s.split())

        if not s.isalpha():
            dfx.loc[id]['Question'] = np.nan
        id+=1


    print(dfx.head(20))
    return(dfx)


# Read url, get contents and scan for search queries
response = requests.get(url)

#print(len(response.content))
soup = BeautifulSoup(response.content, 'html.parser')

top_questions = soup.find_all('div', class_='scroll-box')[0]
topQ = pd.read_html(str(top_questions))[0]

df = pd.DataFrame(topQ)
df.columns = ['Rank',	'Question', 'total Searches', ' CPC', 'keywords']

# Pick only the question column and remove the first row that is not a question
df = df.drop([0])

# Check for queries that have special characters - indicates unwanted text picked from html
df = checkspecialCharc(df)
df = df.dropna()

# Create dataframes to generate input data file in the format required by benchmark
df1 = pd.DataFrame(columns=['Conversation ID',	'Query',	'Expected entities',	'Expected intent',	'Expected awaiting_field'])
df1['Query'] = df['Question']
df1['Expected intent'] = 'q&a'
df1['Conversation ID'] = ""
df1['Expected entities'] = ""
df1['Expected awaiting_field'] = ""

df2 = df1.copy()
df2['Query'] = 'Question'
df2['Expected intent'] = 'q&a'
df2['Conversation ID'] = df2.index
df2['Expected entities'] = ""
df2['Expected awaiting_field'] = 'query'

df_merged = pd.concat([df1, df2]).sort_index(kind='merge')
df_merged.to_excel(outfile, index=False)
print('Done writing to outfile')





