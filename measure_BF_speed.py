import math
import mmh3
import random
from bitarray import bitarray
import pandas as pd
import argparse
import time

# コマンドライン引数の解析
parser = argparse.ArgumentParser(description='Bloom Filter Evaluation Speed Test')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name (CSV file)')
parser.add_argument('--fpr', type=float, required=True, help='False Positive Rate (in %)')
parser.add_argument('--eval_count', type=int, required=True, help='Number of samples to evaluate (N)')
args = parser.parse_args()

class BloomFilter(object):
    '''
    Class for Bloom filter, using murmur3 hash function
    '''
    def __init__(self, items_count, fp_prob):
        self.fp_prob = fp_prob
        self.size = self.get_size(items_count, fp_prob)
        self.hash_count = self.get_hash_count(self.size, items_count)
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)

    def add(self, item):
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            self.bit_array[digest] = True

    def check(self, item):
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if self.bit_array[digest] == False:
                return False
        return True

    @classmethod
    def get_size(self, n, p):
        m = -(n * math.log(p))/(math.log(2)**2)
        return int(m)

    @classmethod
    def get_hash_count(self, m, n):
        k = (m/n) * math.log(2)
        return int(k)

def process_smiles(file_path, dataset_name, fp_prob, eval_count):
    data = pd.read_csv(file_path)
    dataset_name = f"{dataset_name} Dataset"

    positive_data = data[data['ACTIVE'] == 1]['SMILES'].tolist()
    negative_data = data[data['ACTIVE'] == 0]['SMILES'].tolist()
    
    # 挿入対象データと評価対象データの数を出力
    print(f"挿入対象データの数: {len(positive_data)}")
    print(f"評価対象データの数: {len(negative_data)}")
    print(f"使用する評価データ数 (N): {eval_count}\n")

    random.shuffle(positive_data)
    random.shuffle(negative_data)

    # Bloom Filterの作成と評価
    bloom_filter = BloomFilter(len(positive_data), fp_prob / 100)  # FPRは小数として渡す

    # Positive dataをBloom Filterに追加
    for smiles in positive_data:
        bloom_filter.add(smiles)

    # 評価対象のデータからN件ランダムにサンプリング
    sample_negative_data = random.sample(negative_data, eval_count)

    # 評価時の速度を計測
    start_time = time.time()
    false_positives = sum(1 for smiles in sample_negative_data if bloom_filter.check(smiles))
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 実際の偽陽性率を計算
    observed_fpr = (false_positives / eval_count) * 100  # パーセンテージに変換

    # メモリ使用量の計算
    bloom_filter_size_bytes = bloom_filter.size / 8
    bloom_filter_size_kb = bloom_filter_size_bytes / 1024
    bloom_filter_size_mb = bloom_filter_size_kb / 1024

    # ログに結果を出力
    print(f"False Positive Rate (Observed): {observed_fpr:.4f}%")
    print(f"Bloom Filter Size: {bloom_filter_size_kb:.2f} KB ({bloom_filter_size_mb:.2f} MB)")
    print(f"Evaluation Time: {elapsed_time:.4f} seconds\n")

# ファイルパスを設定してBloom Filter評価を実行
dataset = args.dataset
filepath = f'./datasets/{dataset}.csv'
dataset_name = dataset
process_smiles(filepath, dataset_name, args.fpr, args.eval_count)
