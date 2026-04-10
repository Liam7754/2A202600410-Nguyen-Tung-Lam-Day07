import sys
import os
sys.path.append(os.getcwd())
from src.chunking import ChunkingStrategyComparator
from pathlib import Path

comp = ChunkingStrategyComparator()
text = Path('data/team_selection_custom_data.txt').read_text(encoding='utf-8')
results = comp.compare(text, chunk_size=500)
for k, v in results.items():
    print(f"{k}: count={v['count']}, avg={v['avg_length']:.2f}")
