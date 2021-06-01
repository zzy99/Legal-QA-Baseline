import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import rouge
from transformers import MT5ForConditionalGeneration, BertTokenizer, get_cosine_schedule_with_warmup, \
    get_constant_schedule_with_warmup
import jieba
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold

jieba.setLogLevel(20)

rouge = rouge.Rouge()


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

train_df = pd.read_csv('train.csv')[['text', 'answer']]
test_df = pd.read_csv('test.csv')[['text', 'answer']]

model_path = 't5_pegasus_torch'
max_q_len = 512
max_a_len = 150
batch_size = 8
epochs = 15
lr = 1e-4


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


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    # source, target = ' '.join(source), ' '.join(target)
    source, target = ' '.join(jieba.lcut(source)), ' '.join(jieba.lcut(target))
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


folds = KFold(n_splits=10, shuffle=True, random_state=42).split(np.arange(train_df.shape[0]))

cv = []

for fold, (trn_idx, val_idx) in enumerate(folds):
    if fold != 0:
        continue

    train = train_df.loc[trn_idx]
    val = train_df.loc[val_idx]

    train_set = MyDataset(train)
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_set = MyDataset(val)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, collate_fn=collate_fn, shuffle=False)

    model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scaler = GradScaler()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scheduler = get_constant_schedule_with_warmup(optimizer, len(train_loader))

    best = 0

    for epoch in range(epochs):
        tk = tqdm(train_loader, total=len(train_loader))
        losses = AverageMeter()
        model.train()
        for step, batch in enumerate(tk):
            q_id, q_mask, a_id, a_mask = [x.to(device) for x in batch]
            mask = a_mask[:, 1:].reshape(-1).bool()

            prob = model(input_ids=q_id, attention_mask=q_mask, decoder_input_ids=a_id, decoder_attention_mask=a_mask)[
                0]
            prob = prob[:, :-1].reshape((-1, prob.size(-1)))[mask]
            labels = a_id[:, 1:].reshape(-1)[mask]
            loss = loss_fct(prob, labels)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            losses.update(loss.item(), q_id.size(0))
            tk.set_postfix(loss=losses.avg)

        model.eval()
        gens = []
        summaries = []
        for batch in tqdm(val_loader):
            q_id, q_mask, a_id, a_mask = [x.to(device) for x in batch]
            gen = model.generate(max_length=max_a_len, min_length=5, eos_token_id=tokenizer.sep_token_id,
                                 decoder_start_token_id=tokenizer.cls_token_id, num_beams=3, num_beam_groups=3,
                                 input_ids=q_id, attention_mask=q_mask)
            gen = gen[:, 1:].cpu().numpy()
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [s.replace(' ', '') for s in gen]
            gens.extend(gen)

            answer = a_id[:, 1:].cpu().numpy()
            answer = tokenizer.batch_decode(answer, skip_special_tokens=True)
            answer = [s.replace(' ', '') for s in answer]
            summaries.extend(answer)

        scores = compute_rouges(gens, summaries)
        print(scores)
        score = 0.2 * scores['rouge-1'] + 0.3 * scores['rouge-2'] + 0.5 * scores['rouge-l']
        print(score)
        if score > best:
            best = score
            torch.save(model, 'model_fold{}.pt'.format(fold))
