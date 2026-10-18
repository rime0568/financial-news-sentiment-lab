from pathlib import Path
import csv

file_path = Path("data/raw/Stock_news/All_external.csv")

print("File exists:", file_path.exists(), flush=True)
print("File path:", file_path, flush=True)

with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
    reader = csv.reader(f)

    header = next(reader)
    print("\nColumns:", flush=True)
    for col in header:
        print(col, flush=True)

    print("\nFirst 5 rows:", flush=True)
    for i, row in zip(range(5), reader):
        print(row[:10], flush=True)