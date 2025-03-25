import time
from tqdm import tqdm
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision用
import gc  # ガベージコレクション用
from pre_filter import UnifiedGRUModel, MinGRUModel
from pre_filter_parallel import MinGRUModelParallel

def train_model(
    dataset_name,
    X_train,
    y_train,
    vocab_size,
    max_seq_length,
    embedding_dim,
    hidden_size,
    lr=0.0001,
    epochs=50,
    numerical_feature_dim=0,
    model=None,
    optimizer=None,
    start_epoch=0,
    device='cuda',
    model_type='UnifiedGRU',
    batch_size=32
):
    if model is None:
        if model_type == 'MinGRU':
            model = MinGRUModel(
                vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                numerical_feature_dim=numerical_feature_dim
            )
        elif model_type == 'UnifiedGRU':
            model = UnifiedGRUModel(
                vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                bidirectional=False,
                numerical_feature_dim=numerical_feature_dim
            )
        elif model_type == "MinGRUParallel":
            model = MinGRUModelParallel(
                vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                numerical_feature_dim=numerical_feature_dim
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    model.to(device)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # 損失関数: FP16対応のため BCEWithLogitsLoss を使用
    criterion = nn.BCEWithLogitsLoss()

    if isinstance(X_train, tuple):
        X_train_padded, X_train_numerical = X_train
        train_data = TensorDataset(
            torch.tensor(X_train_padded, dtype=torch.long),
            torch.tensor(X_train_numerical, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
    else:
        train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.float32)
        )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    print(f"Starting training for {epochs} epochs...")

    # Mixed Precision用のスケーラー
    scaler = GradScaler()

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        print(f"Before epoch {epoch+1} - GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        total_loss = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{start_epoch + epochs}") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                if numerical_feature_dim > 0 and isinstance(X_train, tuple):
                    X_batch, X_num_batch, y_batch = batch
                    X_batch, X_num_batch, y_batch = (
                        X_batch.to(device),
                        X_num_batch.to(device),
                        y_batch.to(device)
                    )
                else:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    X_num_batch = None

                optimizer.zero_grad()

                # Mixed Precision Training (FP16)
                with autocast():
                    outputs = model(X_batch, X_num_batch) if X_num_batch is not None else model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch.float())

                # スケーラーで勾配計算と更新
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                pbar.update(1)

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{start_epoch + epochs}, Loss: {avg_loss:.6f}')
        torch.cuda.empty_cache()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = f'saved_model/{dataset_name}/{dataset_name}_checkpoint_{timestamp}.pth'
    print(f"Saved model: {checkpoint_path}")
    model_cpu = model.to('cpu')
    torch.save({
        'epoch': start_epoch + epochs,
        'model_state_dict': model_cpu.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    model.to(device)

    return model
