from src.data.processing import compute_dataset_snr_from_files

predictor_path = "/home/nate/Desktop/NFTool/dataset/Predictors_2025-04-15_10-43_Hold-2.csv"
target_path = "/home/nate/Desktop/NFTool/dataset/9_10_24_Hold_02_targets.csv"

snr, snr_db = compute_dataset_snr_from_files(predictor_path, target_path, ridge_eps=1.0)
print(f"New SNR: {snr:.4f} ({snr_db:.2f} dB)")

