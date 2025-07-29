import csv
from pathlib import Path

def load_ground_truth(csv_path):
    ground_truth = {}
    current_device = None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not any(row):
                continue  # skip empty rows

            if row[0].strip().lower() == 'device':
                current_device = row[1].strip()
                ground_truth[current_device] = []

            elif row[0].strip().lower() == 'ground truth' and current_device:
                for i, label in enumerate(row[1:], start=1):
                    label = label.strip().lower()
                    if label in ['connect', 'disconnect']:
                        ground_truth[current_device].append((i, label))

    return ground_truth

ground_path = "logs/ground_truth/groundtruth.csv"
GROUND_TRUTH = load_ground_truth(ground_path)

print(GROUND_TRUTH)
