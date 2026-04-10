import sys
import os
sys.path.append(os.getcwd())

from src.embeddings import MockEmbedder
from src.chunking import compute_similarity

embedder = MockEmbedder()

pairs = [
    ("Học máy rất thú vị", "Machine learning is fun"),
    ("Tôi yêu lập trình", "Tôi thích viết mã"),
    ("Mặt trời mọc ở đằng đông", "Tôi đang ăn phở"),
    ("Ngày mai trời mưa", "Ngày mai có mưa"),
    ("Chào bạn", "Chào bạn")
]

print(f"{'No':<3} | {'Score':<10} | {'Pair'}")
print("-" * 50)
for i, (a, b) in enumerate(pairs, 1):
    vec_a = embedder(a)
    vec_b = embedder(b)
    score = compute_similarity(vec_a, vec_b)
    print(f"{i:<3} | {score:<10.4f} | {a} vs {b}")
