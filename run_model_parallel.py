import argparse
import os
import platform
import torch
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bloom_filter_utils import create_bloom_filter
from train import train_model
import time
from pre_filter_parallel import MinGRUModel, MinGRUModelParallel

def measure_inference_time(X_test, y_test, model, device, repeat=100, batch_size=1024):
    """
    バッチ推論に対応した推論速度測定（進捗バー付き、モデル推論のみ、1サンプルあたりの推論時間を計算）。
    Args:
        X_test: テストデータ（モデル入力形式: (X_test_padded, X_test_numerical) or X_test_padded）。
        y_test: テストラベル。
        model: 学習済みモデル。
        device: 使用するデバイス（CPU/GPU）。
        repeat: 時間計測の繰り返し数。
        batch_size: 推論時のバッチサイズ。
    Returns:
        dict: 推論時間の結果 (平均全体時間、1サンプルあたり時間)。
    """
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    # モデル推論時間を測定
    model_times = []
    total_samples = len(y_test)  # サンプル数

    for _ in tqdm(range(repeat), desc="Overall Inference Time Measurement"):
        start_time = time.time()

        # ---------------------------
        # モデル推論
        # ---------------------------
        model_start = time.time()
        model.eval()
        with torch.no_grad():
            # DataLoader を作成して推論
            if isinstance(X_test, tuple):
                X_padded, X_numerical = X_test
                dataset = TensorDataset(
                    torch.tensor(X_padded, dtype=torch.long),
                    torch.tensor(X_numerical, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.float32)
                )
            else:
                dataset = TensorDataset(
                    torch.tensor(X_test, dtype=torch.long),
                    torch.tensor(y_test, dtype=torch.float32)
                )

            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for batch in tqdm(test_loader, desc="Model Inference", leave=False):
                if isinstance(X_test, tuple):
                    x_padded_batch, x_num_batch, _ = batch
                    x_padded_batch = x_padded_batch.to(device)
                    x_num_batch = x_num_batch.to(device)
                    _ = model(x_padded_batch, x_num_batch)  # 出力は使わず時間計測のみ
                else:
                    x_padded_batch, _ = batch
                    x_padded_batch = x_padded_batch.to(device)
                    _ = model(x_padded_batch)
        model_end = time.time()

        # モデル推論時間を記録
        model_times.append(model_end - model_start)

    # 全体時間の平均
    avg_model_time = np.mean(model_times)
    # 1サンプルあたりの推論時間
    avg_time_per_sample = avg_model_time / total_samples

    return {
        "Model Inference Time (avg)": avg_model_time,
        "Time per sample (avg)": avg_time_per_sample,
    }



def get_false_negatives(X_smiles, X, y, model, batch_size=1024):
    """
    バッチ推論で偽陰性を抽出。
    Args:
        X_smiles: 元の SMILES データ（リスト形式）。
        X: モデル入力形式（数値特徴量がある場合は tuple）。
        y: ラベルデータ。
        model: 学習済みモデル。
        batch_size: バッチサイズ。
    Returns:
        list: 偽陰性の SMILES データ。
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    model.eval()
    device = next(model.parameters()).device
    false_negatives = []

    if isinstance(X, tuple):
        X_padded, X_numerical = X
        dataset = TensorDataset(
            torch.tensor(X_padded, dtype=torch.long),
            torch.tensor(X_numerical, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
    else:
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32)
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(X, tuple):
                x_padded_batch, x_num_batch, y_batch = batch
                x_padded_batch = x_padded_batch.to(device)
                x_num_batch = x_num_batch.to(device)
                outputs = model(x_padded_batch, x_num_batch).squeeze().cpu().numpy()
            else:
                x_padded_batch, y_batch = batch
                x_padded_batch = x_padded_batch.to(device)
                outputs = model(x_padded_batch).squeeze().cpu().numpy()

            predictions.extend(outputs)

    for i, (pred, label) in enumerate(zip(predictions, y)):
        if pred < 0.5 and label == 1:
            false_negatives.append(X_smiles[i])

    return false_negatives


def set_seed(seed_value=42):
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
set_seed(42)

# データ読み込み関数
def load_data(dataset, selected_features=None, sample_fraction=1.0):
    """
    指定されたデータセットを読み込み、必要に応じて特徴量やサンプリングを適用する。
    Args:
        dataset (str): データセットのファイル名（拡張子なし）。
        selected_features (list): 使用する特徴量のリスト（None の場合は全て使用）。
        sample_fraction (float): サンプリングする割合 (0.0 < fraction <= 1.0)。
    Returns:
        tuple: SMILES 配列, 数値特徴量, ラベル。
    """
    file_path = os.path.join("datasets", f"{dataset}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    data = pd.read_csv(file_path)

    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    X_smiles = data['SMILES'].values
    y = data['ACTIVE'].values

    if selected_features:
        try:
            numerical_features = data[selected_features].values
        except KeyError:
            raise ValueError(f"指定された特徴量がデータセットに含まれていません: {selected_features}")
    else:
        numerical_features = data.drop(columns=['SMILES', 'ACTIVE']).values

    if numerical_features.shape[1] == 0:
        numerical_features = None

    return X_smiles, numerical_features, y

# Bloom Filterのサイズ計算
def calculate_bloom_filter_size(bloom_filter):
    bf_size_kb = bloom_filter.bit_array_size_in_bytes() / 1024
    bf_size_mb = bf_size_kb / 1024
    return bf_size_kb,bf_size_mb

# RNNモデルのサイズ計算
def calculate_model_size(model):
    model_size_bytes = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4
    model_size_kb = model_size_bytes / 1024
    model_size_mb = model_size_kb / 1024
    return model_size_bytes, model_size_kb, model_size_mb


# 偽陽性率を計算
def calculate_false_positive_rate(X_test_smiles, y_test, bloom_filter):
    # Negative samples (label = 0) を取得
    negative_samples = [X_test_smiles[i] for i in range(len(y_test)) if y_test[i] == 0]

    # numpy.ndarray -> list of strings
    if isinstance(negative_samples, np.ndarray):
        negative_samples = [str(sample) for sample in negative_samples]
    else:
        negative_samples = [str(sample) for sample in negative_samples]

    # 偽陽性を計算
    false_positives = sum(1 for sample in negative_samples if bloom_filter.check(sample))
    false_positive_rate = false_positives / len(negative_samples) if negative_samples else 0
    return false_positive_rate

# JSON形式で結果を保存する関数
def save_results_to_json(metadata, results, directory="results", filename="results_all_sorted.json"):
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # ファイルパスを生成
    filepath = os.path.join(directory, filename)
    
    # ファイルが存在する場合は既存データを読み込む
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # 新しい結果を追加
    entry = {**metadata, **results, "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    data.append(entry)
    
    # ファイルに書き込み
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {filepath}")


# SMILES文字列のトークナイズとパディング
def tokenize_and_pad_smiles(X_train, X_test, vocab_size, max_seq_length):
    tokenizer = Tokenizer(num_words=vocab_size, char_level=True, oov_token='UNK')
    tokenizer.fit_on_texts(X_train)

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_seq_length, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_seq_length, padding='post')

    return X_train_padded, X_test_padded, tokenizer

def preprocess_data(args, overwrite=False):
    save_dir = "data/"  # 保存先ディレクトリ
    os.makedirs(save_dir, exist_ok=True)

    preprocessed_path = os.path.join(save_dir, f"preprocessed_data_{args.dataset}.npz")
    tokenizer_path = os.path.join(save_dir, f"tokenizer_{args.dataset}.json")

    if not overwrite and os.path.exists(preprocessed_path) and os.path.exists(tokenizer_path):
        print("Loading preprocessed data and tokenizer...")
        data = np.load(preprocessed_path)
        
        # 修正部分: JSONファイルから手動でTokenzierを復元
        with open(tokenizer_path, "r") as f:
            tokenizer_data = json.load(f)
        
        tokenizer = Tokenizer(num_words=1000, char_level=True, oov_token='UNK')
        tokenizer.word_index = tokenizer_data.get("word_index", {})
        tokenizer.index_word = tokenizer_data.get("index_word", {})

        return (
            data["X_train_padded"], data["X_test_padded"],
            data["X_train_numerical"], data["X_test_numerical"],
            data["y_train"], data["y_test"], tokenizer
        )

    print("Processing data for the first time...")
    # データ読み込み
    X_smiles, X_numerical, y = load_data(
        args.dataset, selected_features=args.selected_features, sample_fraction=args.sample_fraction
    )

    X_train_smiles, X_test_smiles, X_train_numerical, X_test_numerical, y_train, y_test = train_test_split(
        X_smiles, X_numerical if X_numerical is not None else np.zeros((len(X_smiles), 0)), y,
        test_size=0.2, random_state=42
    )

    # 数値データのスケーリング
    scaler = StandardScaler()
    if X_train_numerical.shape[1] > 0:
        X_train_numerical = scaler.fit_transform(X_train_numerical)
        X_test_numerical = scaler.transform(X_test_numerical)

    # トークナイズとパディング
    X_train_padded, X_test_padded, tokenizer = tokenize_and_pad_smiles(
        X_train_smiles, X_test_smiles, vocab_size=1000, max_seq_length=args.max_seq_length
    )

    # 保存
    np.savez_compressed(preprocessed_path,
                        X_train_padded=X_train_padded,
                        X_test_padded=X_test_padded,
                        X_train_numerical=X_train_numerical,
                        X_test_numerical=X_test_numerical,
                        y_train=y_train, y_test=y_test)
    with open(tokenizer_path, "w") as f:
        f.write(tokenizer.to_json())
    print(f"Preprocessed data saved to {preprocessed_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

    return X_train_padded, X_test_padded, X_train_numerical, X_test_numerical, y_train, y_test, tokenizer

import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help="Dataset name without extension (e.g., 'HIV')")
    parser.add_argument('--max_seq_length', type=int, default=50)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--bf_fp_prob', type=float, default=0.01)
    parser.add_argument('--bidirectional', action='store_true', help="Use bidirectional GRU")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite preprocessed data")
    args = parser.parse_args()

    # wandbのプロジェクト名設定
    fpr_str = f"{int(args.bf_fp_prob * 1000):03d}"
    project_name = f"{args.dataset}_MinGRU_MinGRUParallel_h{args.hidden_size}_{fpr_str}_{datetime.now().strftime('%Y%m%d')}"

    # wandb初期化
    wandb.init(project=project_name, name="LBF-Experiment")

    # wandbの設定
    wandb.config.update(args)

    # デバイス情報
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print("==== Starting Preprocessing ====")
    # データ前処理
    X_train_padded, X_test_padded, X_train_numerical, X_test_numerical, y_train, y_test, tokenizer = preprocess_data(args, args.overwrite)
    print("==== Preprocessing Complete ====")

    # モデルのトレーニングと評価
    print("==== Starting Training ====")
    X_train_combined = (X_train_padded, X_train_numerical) if X_train_numerical.shape[1] > 0 else X_train_padded
    X_test_combined = (X_test_padded, X_test_numerical) if X_test_numerical.shape[1] > 0 else X_test_padded

    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.float32)
    elif isinstance(y_train, torch.Tensor):
        y_train = y_train.clone().detach().float()

    model = train_model(
        dataset_name=args.dataset,
        X_train=X_train_combined,
        y_train=y_train.to(device),
        vocab_size=1000,
        max_seq_length=args.max_seq_length,
        hidden_size=args.hidden_size,
        lr=args.learning_rate,
        epochs=args.epochs,
        numerical_feature_dim=X_train_numerical.shape[1],
        model_type="MinGRUParallel",
        batch_size=1024,
        device=device
    ).to(device)
    print("==== Training Complete ====")

    # テストデータで偽陰性を抽出
    print("==== Extracting False Negatives ====")
    test_false_negatives = get_false_negatives(X_test_padded, X_test_combined, y_test, model)
    print(f"Number of False Negatives in Test Data: {len(test_false_negatives)}")

    test_false_negatives = [str(item) for item in test_false_negatives]

    print("==== Constructing Bloom Filter ====")
    bloom_filter = create_bloom_filter(test_false_negatives, args.bf_fp_prob)

    # サイズ計算
    print("==== Calculating Sizes ====")
    bf_size_kb, bf_size_mb = calculate_bloom_filter_size(bloom_filter)
    model_size_bytes, model_size_kb, model_size_mb = calculate_model_size(model)
    lbf_total_size_kb = bf_size_kb + model_size_kb
    lbf_total_size_mb = lbf_total_size_kb / 1024

    print(f"Bloom Filter Size: {bf_size_kb:.2f} KB ({bf_size_mb:.6f} MB)")
    print(f"Model Size: {model_size_kb:.2f} KB ({model_size_mb:.6f} MB)")
    print(f"Learned Bloom Filter Total Size: {lbf_total_size_kb:.2f} KB ({lbf_total_size_mb:.6f} MB)")


    # 推論速度の測定 (バッチ推論で行いたい場合は、同様に修正が必要)
    inference_results = measure_inference_time(
        X_test_combined,
        y_test,
        model,
        device,
        repeat=5,  # 繰り返し回数
        batch_size=1024  # 推論バッチサイズ
    )

    print("Inference Time Results:")
    print(json.dumps(inference_results, indent=4))
    # wandbにログ
    wandb.log(inference_results)

    # 偽陽性率を計算
    print("==== Calculating False Positive Rate ====")
    false_positive_rate = calculate_false_positive_rate(X_test_padded, y_test, bloom_filter)
    print(f"False Positive Rate: {false_positive_rate:.6f}")

    wandb.log({
        "Bloom Filter Size (KB)": bf_size_kb,
        "Bloom Filter Size (MB)": bf_size_mb,
        "RNN Model Size (KB)": model_size_kb,
        "RNN Model Size (MB)": model_size_mb,
        "LBF Total Size (KB)": lbf_total_size_kb,
        "LBF Total Size (MB)": lbf_total_size_mb,
        "False Positive Rate": false_positive_rate,
        "False Negatives Count": len(test_false_negatives)
    })

    print("==== Saving Results ====")
    metadata = {
        "Python Version": platform.python_version(),
        "Dataset": args.dataset,
        "Max Sequence Length": args.max_seq_length,
        "Hidden Size": args.hidden_size,
        "Epochs": args.epochs,
        "Learning Rate": args.learning_rate,
        "Training Model Type": "MinGRU",
        "inference Model Type": "MinGRUParallel",
        "BF False Positive Probability": args.bf_fp_prob
    }

    results = {
        "Bloom Filter Size (KB)": bf_size_kb,
        "Bloom Filter Size (MB)": bf_size_mb,
        "RNN Model Size (KB)": model_size_kb,
        "RNN Model Size (MB)": model_size_mb,
        "LBF Total Size (KB)": lbf_total_size_kb,
        "LBF Total Size (MB)": lbf_total_size_mb,
        "False Positive Rate": false_positive_rate,
    }

    save_results_to_json(metadata, results)
    print("Results have been saved.")

    print("==== Process Complete ====")


if __name__ == "__main__":
    main()
