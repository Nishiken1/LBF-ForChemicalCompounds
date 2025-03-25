from backup_filter import CustomBloomFilter
import math

def create_bloom_filter(false_negatives, bf_fp_prob):
    n = len(false_negatives)
    p = bf_fp_prob

    if n == 0:
        n = 1  # Avoid division by zero
    m = int(- (n * math.log(p)) / (math.log(2) ** 2))
    k = max(1, int((m / n) * math.log(2)))

    custom_bloom_filter = CustomBloomFilter(n, p)

    for fn in false_negatives:
        custom_bloom_filter.add(fn)

    return custom_bloom_filter
