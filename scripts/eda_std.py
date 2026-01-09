import csv
import math
from collections import Counter

def calculate_mean(data):
    return sum(data) / len(data)

def calculate_std(data, mean):
    return math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))

def ascii_histogram(data, bins=20, width=50):
    if not data: return ""
    min_val, max_val = min(data), max(data)
    if min_val == max_val: return f"All values are {min_val}"
    
    bin_width = (max_val - min_val) / bins
    hist = [0] * bins
    for x in data:
        idx = int((x - min_val) / bin_width)
        if idx >= bins: idx = bins - 1
        hist[idx] += 1
    
    max_count = max(hist)
    output = []
    output.append(f"Distribution (Min: {min_val:.2f}, Max: {max_val:.2f})")
    for i, count in enumerate(hist):
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = '#' * bar_len
        bin_start = min_val + i * bin_width
        bin_end = min_val + (i + 1) * bin_width
        output.append(f"{bin_start:6.2f} - {bin_end:6.2f} | {bar} ({count})")
    return "\n".join(output)

def main():
    print("Loading data...")
    
    numerical_cols = {
        'age': [], 'study_hours': [], 'class_attendance': [], 
        'sleep_hours': [], 'exam_score': []
    }
    categorical_cols = {
        'gender': [], 'course': [], 'internet_access': [], 
        'sleep_quality': [], 'study_method': [], 'facility_rating': [], 'exam_difficulty': []
    }
    
    row_count = 0
    with open('train.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            try:
                for col in numerical_cols:
                    if row[col]:
                        numerical_cols[col].append(float(row[col]))
                for col in categorical_cols:
                    categorical_cols[col].append(row[col])
            except ValueError:
                continue 

    print(f"Total Rows Processed: {row_count}")

    print("\n--- Numerical Statistics ---")
    for col, data in numerical_cols.items():
        if not data: continue
        mean = calculate_mean(data)
        std = calculate_std(data, mean)
        mini = min(data)
        maxi = max(data)
        print(f"\n{col.upper()}:")
        print(f"  Count: {len(data)}")
        print(f"  Mean:  {mean:.4f}")
        print(f"  Std:   {std:.4f}")
        print(f"  Min:   {mini:.4f}")
        print(f"  Max:   {maxi:.4f}")

    print("\n--- Categorical Statistics ---")
    for col, data in categorical_cols.items():
        print(f"\n{col.upper()}:")
        counts = Counter(data)
        for k, v in counts.most_common():
            print(f"  {k}: {v} ({v/len(data)*100:.1f}%)")

    print("\n--- Target Distribution (exam_score) ---")
    print(ascii_histogram(numerical_cols['exam_score']))

    print("\n--- Correlations with Exam Score ---")
    # Only calculate if lengths match (which they should if no parsing errors)
    target_data = numerical_cols['exam_score']
    n = len(target_data)
    target_mean = calculate_mean(target_data)
    target_std = calculate_std(target_data, target_mean)
    
    for col in numerical_cols:
        if col == 'exam_score': continue
        data = numerical_cols[col]
        if len(data) != n: 
            print(f"Skipping {col} due to length mismatch")
            continue
        
        col_mean = calculate_mean(data)
        col_std = calculate_std(data, col_mean)
        
        covariance = sum((data[i] - col_mean) * (target_data[i] - target_mean) for i in range(n)) / n
        correlation = covariance / (col_std * target_std) if (col_std * target_std) != 0 else 0
        print(f"  {col}: {correlation:.4f}")

if __name__ == "__main__":
    main()
