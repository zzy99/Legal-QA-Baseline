import json
import os
import pandas as pd
from tqdm import tqdm

files = os.listdir('train')
data = []
for file in tqdm(files):
    with open('train/' + file, encoding='utf-8')as f:
        data.append(json.load(f))
train_df = pd.DataFrame(data)

files = os.listdir('test')
data = []
for file in tqdm(files):
    with open('test/' + file, encoding='utf-8')as f:
        data.append(json.load(f))
test_df = pd.DataFrame(data)

train_df['candidate_answer'] = train_df['candidate_answer'].apply(lambda x: '[SEP]'.join(x))
test_df['candidate_answer'] = test_df['candidate_answer'].apply(lambda x: '[SEP]'.join(x))

train_df['text'] = train_df['question'] + '[SEP]' + train_df['candidate_answer']
test_df['text'] = test_df['question'] + '[SEP]' + test_df['candidate_answer']

test_df['id'] = range(16209, 16209 + len(test_df))
test_df['answer'] = ''

print(train_df, test_df)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)


