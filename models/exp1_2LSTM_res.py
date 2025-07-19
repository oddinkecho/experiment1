import torch
import torch.nn as nn

class ResLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        self.res_proj1 = nn.Linear(embedding_dim, hidden_dim * self.num_directions)
        #处理形状
        
        # 3个单层LSTM堆叠
        self.lstm1 = nn.LSTM(
            input_size=hidden_dim * self.num_directions,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=self.bidirectional,
            dropout=0
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * self.num_directions,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=self.bidirectional,
            dropout=0
        )
        self.lstm3 = nn.LSTM(
            input_size=hidden_dim * self.num_directions,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=self.bidirectional,
            dropout=0
        )
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)  # [seq_len, batch, embedding_dim]
        embedded = self.dropout(embedded)

        #提前处理形状
        embedded = self.res_proj1(embedded) 

        # 第一层 LSTM
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), enforce_sorted=False)
        packed_out1, (hidden1, cell1) = self.lstm1(packed_emb)
        out1, _ = nn.utils.rnn.pad_packed_sequence(packed_out1)  # [seq_len, batch, hidden_dim*2]

        # 第一层残差
        res1 = embedded
        out1 = out1 + res1
        out1 = self.dropout(out1)

        # 第二层 LSTM
        packed_out1 = nn.utils.rnn.pack_padded_sequence(out1, text_lengths.cpu(), enforce_sorted=False)
        packed_out2, (hidden2, cell2) = self.lstm2(packed_out1)
        out2, _ = nn.utils.rnn.pad_packed_sequence(packed_out2)  # [seq_len, batch, hidden_dim*2]

        # 第二层残差
        res2 = out1
        out2 = out2 + res2
        out2 = self.dropout(out2)

        # 第三层 LSTM
        packed_out2 = nn.utils.rnn.pack_padded_sequence(out2, text_lengths.cpu(), enforce_sorted=False)
        packed_out3, (hidden3, cell3) = self.lstm3(packed_out2)
        out3, _ = nn.utils.rnn.pad_packed_sequence(packed_out3)  # [seq_len, batch, hidden_dim*2]

        # 第三层残差（直接相加）
        out3 = out3 + out2
        out3 = self.dropout(out3)

        # 取第三层最后hidden状态拼接双向输出
        final_hidden = torch.cat((hidden3[-2], hidden3[-1]), dim=1)  # [batch, hidden_dim*2]
        final_hidden = self.dropout(final_hidden)

        return self.fc(final_hidden)
