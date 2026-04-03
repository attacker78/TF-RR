import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import find_peaks, butter, filtfilt, argrelextrema
from scipy import signal
from scipy.stats import pearsonr
from evaluations import calculate_pearson_correlation
import pandas as pd
import os


def calculate_cosine_similarity(y_true, y_pred):
    """
    Calculate cosine similarity

    Formula: (y_true · y_pred) / (||y_true|| * ||y_pred||)

    Parameters:
    y_true: Ground truth values array
    y_pred: Predicted values array

    Returns:
    Cosine similarity value
    """
    # Calculate dot product
    dot_product = np.dot(y_true, y_pred)

    # Calculate L2 norm
    norm_true = np.linalg.norm(y_true)
    norm_pred = np.linalg.norm(y_pred)

    # Avoid division by zero
    if norm_true == 0 or norm_pred == 0:
        return 0

    # Calculate cosine similarity
    cosine_sim = dot_product / (norm_true * norm_pred)

    return cosine_sim


def save_comparison_plot(real_data, predict_data, patient_idx, save_path):
    """
    Save comparison plot of real and predicted values

    Parameters:
    real_data: Ground truth values array
    predict_data: Predicted values array
    patient_idx: Patient index
    save_path: Save path
    """
    plt.figure(figsize=(15, 5))
    plt.plot(real_data, label='Real', alpha=0.8, linewidth=1, color='blue')
    plt.plot(predict_data, label='Predict', alpha=0.8, linewidth=1, color='red')
    plt.title(f'Patient {patient_idx}: Real vs Predict Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save image
    plt.savefig(os.path.join(save_path, f'patient_{patient_idx}_waveform.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close figure, free memory


total_ip_mse = []
total_ip_mae = []
total_ip_pcc = []
total_ip_cos = []  # Store cosine similarity

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set data and output paths
data_dir = os.path.join(current_dir, 'data', 'test_capnobase')
output_dir = os.path.join(current_dir, 'results', 're')

# Create directory for saving images
image_save_path = os.path.join(output_dir, 'waveform_plots')
os.makedirs(image_save_path, exist_ok=True)
print(f"Created image save directory: {image_save_path}")

for i in range(41):
    print(f"Processing patient {i}...")
    patient_ip_mse = []
    patient_ip_mae = []
    patient_ip_pcc = []
    patient_ip_cos = []  # Cosine similarity list for each patient

    path = os.path.join(data_dir, 'test_predictions{}.pkl'.format(i))
    path1 = os.path.join(data_dir, 'test_reals{}.pkl'.format(i))

    with open(path, 'rb') as f:
        predict_train = pickle.load(f)
    with open(path1, 'rb') as f_real:
        real_train = pickle.load(f_real)

    predict_numpy_arrays = [tensor.numpy().flatten() for tensor in predict_train]
    real_numpy_arrays = [tensor.numpy().flatten() for tensor in real_train]

    predict = np.concatenate(predict_numpy_arrays)
    real = np.concatenate(real_numpy_arrays)

    # Save complete waveform comparison plot
    # save_comparison_plot(real, predict, i, image_save_path)

    i_i = 0
    while i_i < len(predict):
        temp_real = real[i_i:i_i + 2000]
        temp_predict = predict[i_i:i_i + 2000]

        ip_mse = mean_squared_error(temp_real, temp_predict)
        ip_mae = mean_absolute_error(temp_real, temp_predict)
        ip_pcc = calculate_pearson_correlation(temp_real, temp_predict)
        ip_cos = calculate_cosine_similarity(temp_real, temp_predict)

        patient_ip_mse.append(ip_mse)
        patient_ip_mae.append(ip_mae)
        patient_ip_pcc.append(ip_pcc)
        patient_ip_cos.append(ip_cos)

        i_i = i_i + 2000

    t_ip_mse = np.mean(patient_ip_mse)
    t_ip_mae = np.mean(patient_ip_mae)
    t_ip_pcc = np.mean(patient_ip_pcc)
    t_ip_cos = np.mean(patient_ip_cos)

    total_ip_mse.append(t_ip_mse)
    total_ip_mae.append(t_ip_mae)
    total_ip_pcc.append(t_ip_pcc)
    total_ip_cos.append(t_ip_cos)

# Add cosine similarity to output CSV file
merged_array_ip = np.column_stack((total_ip_mse, total_ip_mae, total_ip_pcc, total_ip_cos))
df_ip = pd.DataFrame(merged_array_ip, columns=['MSE', 'MAE', 'PCC', 'COS'])
output_path = os.path.join(output_dir, 'c_result_rs.csv')
df_ip.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")