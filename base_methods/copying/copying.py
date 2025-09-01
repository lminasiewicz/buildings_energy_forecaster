import numpy as np

def get_data(filepath: str) -> np.ndarray:
    return np.genfromtxt(filepath, delimiter=";", skip_header=False, dtype=str)


def mae(arr: np.ndarray) -> float:
    return float(np.average(np.array([abs(row[1] - row[0]) for row in arr])))


def mse(arr: np.ndarray) -> float:
    return float(np.average(np.array([pow(row[1] - row[0], 2) for row in arr])))


def copying(arr: np.ndarray, output_filepath: str = ".") -> None:
    result = []
    values = arr[arr[:, 2].astype(np.datetime64).argsort()][:, 1]
    total = len(arr)

    for row in range(len(values[1:])):
        result.append([np.float64(values[row]), np.float64(values[row - 1])])
        print(f"{row+1}/{total}", end="\r")

    result = np.array(result)
    mae_result = mae(result)
    mse_result = mse(result)

    print(mae_result, mse_result, total)
    with open(f"{output_filepath}/copying_results.txt", "w") as f:
        f.write(f"MAE: {mae_result}\nMSE: {mse_result}\nTotal Dataset Size: {total}\nInsufficient Data for predictions: {1}")

        



def main() -> None:
    data = get_data("../../data/processed_data/test_dataset_timestamped.csv")
    copying(data)


if __name__ == "__main__":
    main()