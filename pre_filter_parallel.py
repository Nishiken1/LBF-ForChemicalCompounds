import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------
# 1) MinGRUCell
#    (学習済みモデルと同じ構造: cell というサブモジュールを持つ形に合わせます)
#    - tanh を使用して候補隠れ状態を計算する例
# -------------------------------------------------------
class MinGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinGRUCell, self).__init__()
        # 更新ゲートと候補隠れ状態
        self.linear_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        """
        x:      (batch, input_size)
        h_prev: (batch, hidden_size)
        """
        # 結合
        combined = torch.cat([x, h_prev], dim=1)
        # 更新ゲート
        z_t = torch.sigmoid(self.linear_z(combined))
        # 候補隠れ状態 (tanh)
        h_tilde = torch.tanh(self.linear_h(combined))
        # 最終隠れ状態更新
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t


# -------------------------------------------------------
# 2) MinGRU (シーケンシャル版, Pythonループで逐次計算)
# -------------------------------------------------------
class MinGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(MinGRU, self).__init__()
        self.cell = MinGRUCell(input_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns:
          outputs: (batch, seq_len, hidden_size*(2 if bidirectional else 1))
          h_last:  (batch, hidden_size*(2 if bidirectional else 1))
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        hidden_size = self.cell.linear_z.out_features

        # Forward
        h_forward = torch.zeros(batch_size, hidden_size, device=device)
        outputs_forward = []

        for t in range(seq_len):
            h_forward = self.cell(x[:, t, :], h_forward)
            outputs_forward.append(h_forward.unsqueeze(1))

        outputs_forward = torch.cat(outputs_forward, dim=1)

        if self.bidirectional:
            # Backward
            h_backward = torch.zeros(batch_size, hidden_size, device=device)
            outputs_backward = []
            for t in reversed(range(seq_len)):
                h_backward = self.cell(x[:, t, :], h_backward)
                outputs_backward.append(h_backward.unsqueeze(1))
            # reverse -> cat
            outputs_backward = torch.cat(outputs_backward[::-1], dim=1)
            # 結合
            outputs = torch.cat([outputs_forward, outputs_backward], dim=2)
            h_last = torch.cat([h_forward, h_backward], dim=1)
        else:
            outputs = outputs_forward
            h_last = h_forward

        return outputs, h_last


# -------------------------------------------------------
# 3) MinGRUModel (シーケンシャルGRUモデル: Embedding + MinGRU + FC)
# -------------------------------------------------------
class MinGRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,
                 bidirectional=False, numerical_feature_dim=0):
        super(MinGRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.min_gru = MinGRU(embedding_dim, hidden_size, bidirectional)
        self.dropout = nn.Dropout(0.3)

        fc_input_dim = (2 * hidden_size if bidirectional else hidden_size) + numerical_feature_dim
        self.fc = nn.Linear(fc_input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_numerical=None):
        """
        x: (batch, seq_len)
        x_numerical: (batch, numerical_feature_dim) or None
        """
        # Embedding
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
        # minGRU
        outputs, h_last = self.min_gru(emb)
        h_last = self.dropout(h_last)
        # 数値特徴量があれば結合
        if x_numerical is not None:
            x_numerical = x_numerical.float()
            h_last = torch.cat([h_last, x_numerical], dim=1)
        # 全結合
        out = self.fc(h_last)
        return self.sigmoid(out)


# -------------------------------------------------------
# 4) MinGRUParallel
#    (論文のparallel prefix-scanを擬似的に表現したクラス)
#    - 本来は log(seq_len) 段で計算するが、ここでは実質シーケンシャル
# -------------------------------------------------------
class MinGRUParallel(nn.Module):
    """
    過去の h_{t-1} に依存しない形で z_t, h_tilde を計算する minGRU の並列スキャン版。
    実際にはシーケンシャル実装に近く、Pythonループが残るため高速化は限定的。
    """
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(MinGRUParallel, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # forward用
        self.cell = nn.Module()
        self.cell.linear_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell.linear_h = nn.Linear(input_size + hidden_size, hidden_size)

        # bidirectional用
        if bidirectional:
            self.cell.linear_z_bw = nn.Linear(input_size + hidden_size, hidden_size)
            self.cell.linear_h_bw = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        returns: (batch, seq_len, hidden_size*(2 if bidirectional else 1))
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # Forward pass
        h_forward = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs_forward = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h_forward], dim=1)
            z_t = torch.sigmoid(self.cell.linear_z(combined))
            h_tilde = torch.tanh(self.cell.linear_h(combined))
            h_forward = (1 - z_t) * h_forward + z_t * h_tilde
            outputs_forward.append(h_forward.unsqueeze(1))

        outputs_forward = torch.cat(outputs_forward, dim=1)

        if self.bidirectional:
            h_backward = torch.zeros(batch_size, self.hidden_size, device=device)
            outputs_backward = []
            for t in reversed(range(seq_len)):
                x_t = x[:, t, :]
                comb_bw = torch.cat([x_t, h_backward], dim=1)
                z_t_bw = torch.sigmoid(self.cell.linear_z_bw(comb_bw))
                h_tilde_bw = torch.tanh(self.cell.linear_h_bw(comb_bw))
                h_backward = (1 - z_t_bw) * h_backward + z_t_bw * h_tilde_bw
                outputs_backward.append(h_backward.unsqueeze(1))
            outputs_backward = torch.cat(outputs_backward[::-1], dim=1)
            outputs = torch.cat([outputs_forward, outputs_backward], dim=2)
        else:
            outputs = outputs_forward

        return outputs


# -------------------------------------------------------
# 5) MinGRUModelParallel
#    (Embedding + MinGRUParallel + FC)
# -------------------------------------------------------
class MinGRUModelParallel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,
                 bidirectional=False, numerical_feature_dim=0):
        super(MinGRUModelParallel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.min_gru = MinGRUParallel(embedding_dim, hidden_size, bidirectional)

        out_dim = (2 * hidden_size if bidirectional else hidden_size) + numerical_feature_dim
        self.fc = nn.Linear(out_dim, 1)
        # 注意: ここでSigmoidは呼び出さなくてもOK（BCEWithLogitsLossを使う場合）
        # 研究や実験用なら後段で呼ぶかどうか調整
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_numerical=None):
        """
        x: (batch, seq_len)
        x_numerical: (batch, numerical_feature_dim) or None
        """
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
        h_seq = self.min_gru(emb)  # (batch, seq_len, hidden_size * (2 if bidirectional else 1))

        # 最終タイムステップ
        h_last = h_seq[:, -1, :]
        if x_numerical is not None:
            h_last = torch.cat([h_last, x_numerical], dim=1)

        out = self.fc(h_last)
        out = self.sigmoid(out)
        return out
