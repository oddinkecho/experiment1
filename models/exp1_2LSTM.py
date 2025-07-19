import torch
import torch.nn as nn

class lwLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        # Embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM 层
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

        # 全连接输出层
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text: [sent_len, batch_size]
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        # pack_padded_sequence 处理可变长度
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # 双向 LSTM 拼接最后一个前向 + 反向隐藏状态
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # 输出
        output = self.fc(hidden)
        return output
