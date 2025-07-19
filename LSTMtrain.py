import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models.exp1_2LSTM import lwLSTM
from models.exp1_2LSTM_res import  ResLSTM 
from torch.utils.data import DataLoader
from collections import Counter


# 参数
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1  # 二分类
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.4
BATCH_SIZE = 64

# 文本转token id
def text_pipeline(x):
    return vocab(tokenizer(x))

def label_pipeline(x):
    # 如果是字符串
    if isinstance(x, str):
        x = x.strip().lower()
        return 1 if x == 'pos' else 0
    # 如果是整数
    elif isinstance(x, int):
        return x - 1
    else:
        raise ValueError(f"Unexpected label type: {type(x)}")


# DataLoader批处理
def collate_batch(batch):

    #print("RAW LABEL:", repr(batch[0][0]))
    #print("after pipe label:",label_pipeline(batch[0][0]))


    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.long)
        text_list.append(processed_text)
        lengths.append(len(processed_text))

    text_list = pad_sequence(text_list, padding_value=pad_idx)
    lengths = torch.tensor(lengths, dtype=torch.long)
    label_list = torch.tensor(label_list, dtype=torch.float)
    return text_list.to(device), lengths.to(device), label_list.to(device)

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

# 训练和评估函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for text, text_lengths, labels in dataloader:
        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for text, text_lengths, labels in dataloader:
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer = get_tokenizer('basic_english')
#print("Building vocab")
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

pad_idx = vocab["<pad>"]


# 参数
VOCAB_SIZE = len(vocab)

# 加载数据
train_iter, test_iter = IMDB(split=('train', 'test'))
train_dataset = list(train_iter)
test_dataset = list(test_iter)

#labels = [label for label, text in train_dataset]
#print(Counter(labels))
#print("test_pipeline")
#print(label_pipeline('pos'), label_pipeline('neg'))  # 应该输出 1 0
#print("length_dataset")
#print(len(train_dataset), len(test_dataset))  


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

model = ResLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
               N_LAYERS, BIDIRECTIONAL, DROPOUT, pad_idx).to(device)
#model = lwLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
#               N_LAYERS, BIDIRECTIONAL, DROPOUT, pad_idx).to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters())



# 训练主循环
N_EPOCHS = 20

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_dataloader, criterion)
    print(f'Epoch: {epoch+1}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
