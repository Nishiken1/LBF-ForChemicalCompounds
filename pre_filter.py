import torch
import torch.nn as nn
from minGRU_pytorch import minGRU  # minGRUをインポート

# -------------------------------------------------------
# UnifiedGRUModel: PyTorch標準のGRUを使用する実装
# -------------------------------------------------------
class UnifiedGRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, bidirectional=False, numerical_feature_dim=0):
        super(UnifiedGRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        fc_input_dim = (2 * hidden_size if bidirectional else hidden_size) + numerical_feature_dim
        self.fc = nn.Linear(fc_input_dim, 1)

    def forward(self, x, x_numerical=None):
        x = self.embedding(x)
        outputs, h_last = self.gru(x)

        if self.gru.bidirectional:
            h_last = torch.cat([h_last[-2], h_last[-1]], dim=1)
        else:
            h_last = h_last[-1]

        h_last = self.dropout(h_last)

        if x_numerical is not None:
            h_last = torch.cat((h_last, x_numerical), dim=1)

        x = self.fc(h_last)
        return x


# -------------------------------------------------------
# MinGRUModel: minGRUを使用する実装 (修正箇所)
# -------------------------------------------------------
class MinGRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, bidirectional=False, numerical_feature_dim=0):
        super(MinGRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # ここで hidden_size を使うように修正
        # minGRU_pytorch の実装によっては、input_size / hidden_size 両方を指定できる場合もあります。
        # 単一引数 dim=... しか受け付けないなら、そちらを hidden_size に合わせてください。
        # self.min_gru = minGRU(dim=hidden_size)
        self.min_gru = minGRU(dim=embedding_dim) 

        self.dropout = nn.Dropout(0.3)

        # bidirectional=True は未対応ならエラーにする
        if bidirectional:
            raise NotImplementedError("bidirectional=True は MinGRUModel では未対応です。")

        # minGRU の最終出力次元が hidden_size になるので修正
        # fc_input_dim = hidden_size + numerical_feature_dim
        fc_input_dim = embedding_dim + numerical_feature_dim
        self.fc = nn.Linear(fc_input_dim, 1)

    def forward(self, x, x_numerical=None):
        x = self.embedding(x)               # [batch, seq_len, embedding_dim]
        outputs = self.min_gru(x)           # [batch, seq_len, hidden_size] (最後の次元が hidden_size になる)

        h_last = outputs[:, -1, :]          # 最終タイムステップの出力を取得 (batch, hidden_size)
        h_last = self.dropout(h_last)

        if x_numerical is not None:
            h_last = torch.cat((h_last, x_numerical), dim=1)

        x = self.fc(h_last)
        return x
