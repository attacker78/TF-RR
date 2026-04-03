from scipy.signal import find_peaks, butter, filtfilt
import numpy as np
import pickle
import pandas as pd
from evaluations import calculate_rmse
from sklearn.metrics import mean_absolute_error
import os


def butter_bandpass_filter(data, fs, lowcut=0.1, highcut=0.5, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def find_RR(data_):
    total_RR = []
    index = 0
    while index < len(data_):
        data = data_[index:index + 2000]
        peaks, _ = find_peaks(data)  # Local maximum points on vertical axis
        mins, _ = find_peaks(data * -1)  # Local minimum points on vertical axis
        # Y-coordinates of local maxima, Y-coordinates of local minima
        y_max = data[peaks]
        y_min = data[mins]

        # Convert to list for indexing
        y_max_list = y_max.tolist()
        y_min_list = y_min.tolist()

        av = sorted(y_max)  # Sort from small to large
        Q3 = np.percentile(av, 75)  # Take the third quartile of all local maximum y-coordinates
        threshold = 0.2 * Q3  # Define threshold level
        ########## Determine whether it is a valid respiratory cycle. First check if both maxima are greater than the threshold, then check if there is a minimum less than 0 between them
        RR = 0
        i = 0
        max_index = len(y_max_list) - 1
        while i < max_index:
            if y_max_list[i] > threshold and y_max_list[i + 1] > threshold:
                # a, b are the index values of the two maxima in the original list
                a = peaks[i]
                b = peaks[i + 1]
                # Find minima within the two maxima
                for z in y_min_list:
                    c = (np.where(data == z)[0])[0]
                    if c > a and c < b:
                        if z < 0:
                            RR = RR + 1
            i = i + 1
        index = index + 2000
        total_RR.append(RR)

    return total_RR


if __name__ == '__main__':
    total_rr_mae = []
    total_rr_rmse = []

    # Set paths relative to the current script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data', 'test_capnobase')
    output_dir = os.path.join(current_dir, 'results', 're')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i in range(41):
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

        filter_predict = butter_bandpass_filter(predict, fs=125)
        filter_real = butter_bandpass_filter(real, fs=125)

        i_i = 0

        real_rr = []
        predict_rr = []

        while i_i < len(real) - 2000:  # Fix: len(real)-2000
            real_data = filter_real[i_i:i_i + 2000]
            predict_data = filter_predict[i_i:i_i + 2000]

            try:
                real_rate = find_RR(real_data)
                predict_rate = find_RR(predict_data)
                # find_RR returns a list, need to take the first element (assuming each window returns only one RR value)
                if real_rate and predict_rate:
                    real_rr.append(real_rate[0])
                    predict_rr.append(predict_rate[0])
                i_i = i_i + 2000

            except Exception as e:
                print(f"Error at index {i_i}: {e}")
                i_i = i_i + 2000

        # Ensure both lists have the same length
        min_len = min(len(real_rr), len(predict_rr))
        if min_len > 0:
            real_rr = real_rr[:min_len]
            predict_rr = predict_rr[:min_len]

            mae = mean_absolute_error(real_rr, predict_rr)
            rmse = calculate_rmse(real_rr, predict_rr)

            total_rr_mae.append(mae)
            total_rr_rmse.append(rmse)
            print(f"File {i}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        else:
            print(f"File {i}: No valid data")

    # Merge only MAE and RMSE columns
    merged_array_ip = np.column_stack((total_rr_mae, total_rr_rmse))
    df_rr = pd.DataFrame(merged_array_ip, columns=['MAE', 'RMSE'])
    output_path = os.path.join(output_dir, 'c_result_rr.csv')
    df_rr.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")