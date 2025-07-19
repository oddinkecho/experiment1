from torchtext.datasets import IMDB

train_iter = IMDB(split='train')
for i in range(10):
    print(next(iter(train_iter)))
    train_iter=train_iter+1
