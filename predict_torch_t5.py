import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import rouge
from transformers import MT5ForConditionalGeneration, BertTokenizer
import jieba
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

jieba.setLogLevel(20)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_df = pd.read_csv('test.csv')[['text', 'answer']]
print(test_df)

model_path = 't5_pegasus_torch'
max_q_len = 512
max_a_len = 150
batch_size = 32


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


tokenizer = T5PegasusTokenizer.from_pretrained(model_path)


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.text.values[idx])
        answer = str(self.df.answer.values[idx])
        return text, answer


def collate_fn(data):
    text = tokenizer([x[0] for x in data], padding='max_length', max_length=max_q_len, truncation=True,
                     return_tensors='pt')
    q_id = text['input_ids']
    q_mask = text['attention_mask']
    answer = tokenizer([x[1] for x in data], padding='max_length', max_length=max_a_len, truncation=True,
                       return_tensors='pt')
    a_id = answer['input_ids']
    a_mask = answer['attention_mask']
    return q_id, q_mask, a_id, a_mask


test_set = MyDataset(test_df)
test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8)


def generate(text, max_length=30):
    feature = tokenizer.encode(text, return_token_type_ids=True, return_tensors='pt',
                               max_length=max_q_len, truncation=True)
    feature = {'input_ids': feature}
    feature = {k: v.to(device) for k, v in list(feature.items())}

    gen = model.generate(max_length=max_length, eos_token_id=tokenizer.sep_token_id,
                         decoder_start_token_id=tokenizer.cls_token_id, num_beams=3,
                         **feature).cpu().numpy()[0]
    gen = gen[1:]
    gen = tokenizer.decode(gen, skip_special_tokens=True).replace(' ', '')
    return gen


# model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
model = torch.load('model_fold0.pt').to(device)
model.eval()

preds = []
for batch in tqdm(test_loader):
    # gen = generate(text, max_length=max_a_len)
    q_id, q_mask, a_id, a_mask = [x.to(device) for x in batch]
    gen = model.generate(max_length=max_a_len, eos_token_id=tokenizer.sep_token_id,
                         decoder_start_token_id=tokenizer.cls_token_id, num_beams=3, num_beam_groups=3,
                         input_ids=q_id, attention_mask=q_mask, return_dict_in_generate=True, output_scores=True)

    gen = gen['sequences'].cpu().numpy()
    gen = gen[:, 1:]
    pred = tokenizer.batch_decode(gen, skip_special_tokens=True)
    pred = [s.replace(' ', '') for s in pred]
    preds.extend(pred)

test_df['answer'] = preds

test_df[['id', 'answer']].to_json('tmp.json', orient='records', force_ascii=False)

with open('sub1.json', 'w', encoding='utf-8')as f:
    sub = json.load(open('tmp.json', encoding='utf-8'))
    json.dump(sub, f, ensure_ascii=False, indent=4)
