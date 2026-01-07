import pandas as pd
import numpy as np

def check_leakage(predictor_path, target_path):
    print(f"Checking {predictor_path} and {target_path}")
    
    # Try reading with and without headers to see which makes sense
    df_X_no_header = pd.read_csv(predictor_path, header=None)
    df_y_no_header = pd.read_csv(target_path, header=None)
    
    print(f"X shape (no header): {df_X_no_header.shape}")
    print(f"y shape (no header): {df_y_no_header.shape}")
    
    # Check for NaNs
    print(f"X NaNs: {df_X_no_header.isna().sum().sum()}")
    print(f"y NaNs: {df_y_no_header.isna().sum().sum()}")
    
    y = df_y_no_header.iloc[:, 0].values
    print(f"Target variance: {np.var(y)}")
    print(f"Target std: {np.std(y)}")
    
    # Check if any column in X is suspiciously similar to y
    leaks = []
    for i in range(df_X_no_header.shape[1]):
        col = df_X_no_header.iloc[:, i].values
        # Check correlation
        if len(col) == len(y):
            corr = np.corrcoef(col, y)[0, 1]
            if abs(corr) > 0.99:
                leaks.append((i, corr))
                
    if leaks:
        print(f"FOUND POTENTIAL LEAKS (Correlation > 0.99):")
        for idx, corr in leaks:
            print(f"  Column {idx}: corr={corr}")
    else:
        print("No columns with correlation > 0.99 found.")

    # Check for index leak (target is just a function of row index)
    indices = np.arange(len(y))
    idx_corr = np.corrcoef(indices, y)[0, 1]
    print(f"Index correlation with target: {idx_corr}")

    # Check first few values
    print("\nFirst 5 target values:", y[:5])
    print("First 5 values of first 5 predictor columns:")
    print(df_X_no_header.iloc[:5, :5])

if __name__ == "__main__":
    check_leakage("/home/nate/Desktop/NFTool/dataset/Predictors_2025-04-15_10-43_Hold-2.csv", 
                  "/home/nate/Desktop/NFTool/dataset/9_10_24_Hold_02_targets.csv")

