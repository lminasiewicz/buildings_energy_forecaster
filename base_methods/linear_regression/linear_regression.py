import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error



def get_data(filepath_train: str, filepath_test: str) -> tuple[np.ndarray, np.ndarray]:
    bool_to_floatstr_converter = {11: lambda s: "1.0" if bool(s) else "0.0"}
    train = np.genfromtxt(filepath_train, delimiter=";", skip_header=False, dtype=str, converters=bool_to_floatstr_converter).astype(np.float64)
    test = np.genfromtxt(filepath_test, delimiter=";", skip_header=False, dtype=str, converters=bool_to_floatstr_converter).astype(np.float64)
    return train, test



def linear_regression(train_data: np.ndarray, test_data: np.ndarray, output_filepath: str = ".") -> None:
    target = 1 
    feature_indices = list(range(2, 12))

    x_train = train_data[:, feature_indices]
    y_train = train_data[:, target]
    x_test  = test_data[:, feature_indices]
    y_test  = test_data[:, target]

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred_train = lr.predict(x_train)
    y_pred_test  = lr.predict(x_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_test  = mean_squared_error(y_test, y_pred_test)
    mae_test  = mean_absolute_error(y_test, y_pred_test)

    print(f"Train MAE: {mae_train}, Train MSE: {mse_train}")
    print(f"Test  MAE: {mae_test}, Test  MSE: {mse_test}")

    with open(f"{output_filepath}/linear_regression_results.txt", "w") as f:
        f.write(f"Train MAE: {mae_train}\nTrain MSE: {mse_train}\nTest MAE: {mae_test}\nTest MSE: {mse_test}")



def main() -> None:
    train_data, test_data = get_data("../../data/processed_data/train_dataset.csv", "../../data/processed_data/test_dataset.csv")
    linear_regression(train_data, test_data)



if __name__ == "__main__":
    main()
    