import numpy as np


def get_data(filepath: str) -> np.ndarray:
    return np.genfromtxt(filepath, delimiter=";", skip_header=False, dtype=str)


def mae(arr: np.ndarray) -> float:
    return float(np.average(np.array([abs(row[1] - row[0]) for row in arr])))


def mse(arr: np.ndarray) -> float:
    return float(np.average(np.array([pow(row[1] - row[0], 2) for row in arr])))



def moving_average(data: np.ndarray, timeframe_h: int, output_filepath: str = ".") -> None:
    result = []
    data = data.astype(np.object_)
    total = len(data)
    missing_data = 0

    for row in range(len(data)):
        building_id = data[row, 0]
        upper_bound = np.datetime64(data[row, 2])
        lower_bound =  np.datetime64(data[row, 2]) - np.timedelta64(timeframe_h, "h")

        id_mask = (data[:, 0] == building_id)
        data_c = data[id_mask]
        data_c[:, 2] = data_c[:, 2].astype(np.datetime64)
        upper_bound_mask = (data_c[:, 2] < upper_bound)
        data_c = data_c[upper_bound_mask]
        lower_bound_mask = (data_c[:, 2] > lower_bound)
        data_c = data_c[lower_bound_mask]

        if len(data_c) != 0:
            result.append((np.float64(data[row, 1]), np.average(data_c[:, 1].astype(np.float64))))
        else:
            missing_data += 1
        print(f"{row+1}/{total}", end="\r")
    

    result = np.array(result)
    mae_result = mae(result)
    mse_result = mse(result)

    print(mae_result, mse_result, total, missing_data)
    with open(f"{output_filepath}/moving_average_results.txt", "w") as f:
        f.write(f"MAE: {mae_result}\nMSE: {mse_result}\nTotal Dataset Size: {total}\nInsufficient Data for predictions: {missing_data}")




def main() -> None:
    data = get_data("../../data/processed_data/test_dataset_timestamped.csv")
    moving_average(data, 72)



if __name__ == "__main__":
    main()