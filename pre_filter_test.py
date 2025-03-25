import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from pre_filter import UnifiedGRUModel, MinGRUModel

# ダミーデータの準備
batch_size = 32
seq_len = 1024
vocab_size = 5000
embedding_dim = 64
hidden_size = 128
numerical_feature_dim = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "UnifiedGRU": UnifiedGRUModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        bidirectional=False,
        numerical_feature_dim=numerical_feature_dim
    ),
    "MinGRU": MinGRUModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        bidirectional=False,
        numerical_feature_dim=numerical_feature_dim
    )
}

# 入力データ
x = torch.randint(0, vocab_size, (batch_size, seq_len))
x_numerical = torch.randn(batch_size, numerical_feature_dim)
x, x_numerical = x.to(device), x_numerical.to(device)

for model_name, model in models.items():
    print(f"=== Profiling {model_name} ===")

    model = model.to(device)
    model.eval()

    # ---------------------------
    #  1. profileスコープを単純化
    #  2. scheduleやon_trace_readyを使わない
    # ---------------------------
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for step in range(5):
            with record_function("model_inference"):
                model(x, x_numerical)

    # スコープ外でもprofは有効 (スケジュール未使用時)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("Profiling complete.")
