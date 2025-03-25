import math
import mmh3
from bitarray import bitarray
class CustomBloomFilter(object):
    def __init__(self, items_count, fp_prob):
        self.fp_prob = fp_prob
        if items_count <= 0:  # items_count が 0 以下の場合の対処
            items_count = 1   # ゼロ除算を防ぐために最低値を設定
        self.size = self.get_size(items_count, fp_prob)
        self.hash_count = self.get_hash_count(self.size, items_count)
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)  # 初期状態は空（全て0）

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

    def __contains__(self, item):
        """`in` 演算子で使用するためのメソッド"""
        return self.check(item)

    @classmethod
    def get_size(cls, n, p):
        """必要なビット配列のサイズを計算する"""
        if n <= 0:  # n が 0 以下の場合の対処
            n = 1  # 最小サイズを 1 に設定
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @classmethod
    def get_hash_count(cls, m, n):
        """必要なハッシュ関数の数を計算する"""
        if n <= 0:  # n が 0 以下の場合の対処
            n = 1  # ハッシュ関数の数を 1 に設定
        k = (m / n) * math.log(2)
        return int(k)

    def bit_array_size_in_bytes(self):
        """ビット配列のサイズをバイト単位で返す"""
        return len(self.bit_array) / 8
