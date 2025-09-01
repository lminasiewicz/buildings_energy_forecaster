import numpy as np
from datetime import date
import sys
import math

def strip_dataset(up_to: int, in_filepath: str, out_filepath: str, delimiter: str = ";") -> None:
    """Strip a dataset with building IDs to only contain data for buildings up to up_to ID.
    This is done because temperature data is not available past a certain Building ID.
    Additionally, remove all rows that don't have energy observation values."""
    data = np.genfromtxt(in_filepath, delimiter=delimiter, skip_header=True, dtype=str)
    bool_mask = (data[:, 4] != "") & (data[:, 1].astype(int) <= up_to) # Boolean mask to filter out incompatible Building IDs and empty energy values
    stripped_data = data[bool_mask]
    stripped_data.astype(str)
    np.savetxt(out_filepath, stripped_data, delimiter=delimiter, fmt="%s")
    print(f"Successfully stripped {in_filepath} and saved to {out_filepath}.")


def concat_dataset(d1_filepath: str, d2_filepath: str, out_filepath: str, delimiter: str = ";") -> None:
    """Concatenate 2 datasets of equivalent structure. Mainly created to concatenate the training and test datasets into one for processing."""
    d1 = np.genfromtxt(d1_filepath, delimiter=delimiter, skip_header=False, dtype=str)
    d2 = np.genfromtxt(d2_filepath, delimiter=delimiter, skip_header=False, dtype=str)
    d_all = np.concatenate((d1, d2), axis=0)
    np.savetxt(out_filepath, d_all, delimiter=delimiter, fmt="%s")
    print(f"Successfully concatenated {d1_filepath} with {d2_filepath} and saved to {out_filepath}.")


def remove_redundant_columns(in_filepath: str, cols: list[int], out_filepath: str, delimiter: str = ";") -> None:
    "Removes a list of columns from the dataset."
    data = np.genfromtxt(in_filepath, delimiter=delimiter, skip_header=False, dtype=str)
    data = np.delete(data, cols, axis=1)
    np.savetxt(out_filepath, data, delimiter=delimiter, fmt="%s")
    print(f"Successfully removed columns {cols} from {in_filepath} and saved to {out_filepath}.")


def merge_energy_weather(energy_filepath: str, weather_filepath: str, out_filepath: str, delimiter: str = ";", max_timedelta_hours: int = 12) -> None:
    """Merge a file containing energy consumption data with a file containing temperature data.
    The result will have all relevant fields of both files. It will match points by closest timestamp.
    The timestamp of the energy datapoint takes precedence. 
    max_timedelta_hours defines a maximum number of hours past which datapoints are considered to not be close enough in time to merge, and the data is discarded."""
    energy_data = np.genfromtxt(energy_filepath, delimiter=delimiter, skip_header=False, dtype=str)
    weather_data = np.genfromtxt(weather_filepath, delimiter=delimiter, skip_header=True, dtype=str)

    # Convert columns
    energy_ids = energy_data[:, 0].astype(int)
    energy_times = energy_data[:, 1]
    energy_vals = energy_data[:, 2].astype(float)

    weather_times = weather_data[:, 0]
    weather_temps = weather_data[:, 1].astype(float)
    weather_dists = weather_data[:, 2].astype(float)
    weather_ids = weather_data[:, 3].astype(int)

    energy_times_dt = energy_times.astype("datetime64[m]")
    weather_times_dt = weather_times.astype("datetime64[m]")

    max_diff = np.timedelta64(max_timedelta_hours, "h")

    # Sort energy and weather data by ID_Building, then timestamp
    energy_sort_idx = np.lexsort((energy_times_dt, energy_ids))
    weather_sort_idx = np.lexsort((weather_times_dt, weather_ids))

    energy_ids = energy_ids[energy_sort_idx]
    energy_times_dt = energy_times_dt[energy_sort_idx]
    energy_vals = energy_vals[energy_sort_idx]

    weather_ids = weather_ids[weather_sort_idx]
    weather_times_dt = weather_times_dt[weather_sort_idx]
    weather_temps = weather_temps[weather_sort_idx]
    weather_dists = weather_dists[weather_sort_idx]

    merged = []
    i = j = 0
    n_energy = len(energy_ids)
    n_weather = len(weather_ids)

    while i < n_energy: # O(n log n)
        current_id = energy_ids[i]
        while j < n_weather and weather_ids[j] < current_id:
            j += 1
        
        # Collect weather indices for current building
        w_start = j
        while j < n_weather and weather_ids[j] == current_id:
            j += 1
        w_end = j

        if w_start == w_end: # No weather data 
            i += 1
            continue

        # Get weather data for the building
        w_times = weather_times_dt[w_start:w_end]
        w_temps = weather_temps[w_start:w_end]
        w_dists = weather_dists[w_start:w_end]

        # Find the nearest weather point
        e_idx = i
        while e_idx < n_energy and energy_ids[e_idx] == current_id:
            e_time = energy_times_dt[e_idx]
            insert_pos = np.searchsorted(w_times, e_time) # numpy.searchsorted uses binsearch
            best_idx = -1
            best_diff = max_diff + np.timedelta64(1, "m")

            for offset in [-1, 0]: # Neighbours
                idx = insert_pos + offset
                if 0 <= idx < len(w_times):
                    time_diff = abs(w_times[idx] - e_time)
                    if time_diff <= max_diff and time_diff < best_diff:
                        best_diff = time_diff
                        best_idx = idx

            if best_idx >= 0: # AKA if a point was found at all
                merged.append([
                    str(current_id),
                    f"{energy_vals[e_idx]:.4f}",
                    str(e_time),
                    f"{w_temps[best_idx]:.1f}",
                    f"{w_dists[best_idx]:.2f}"
                ])
            e_idx += 1
        i = e_idx

    merged = np.array(merged)
    np.savetxt(out_filepath, merged, delimiter=delimiter, fmt="%s")
    print(f"Saved {len(merged)} merged rows to {out_filepath}")


def add_weekday_days_off(in_filepath: str, metadata_filepath: str, out_filepath: str, delimiter: str = ";") -> None:
    """Adds the day_off column to a merged dataset and fills it with boolean values based on a weekly schedule preset in a metadata file (from raw data)"""
    data = np.genfromtxt(in_filepath, delimiter=delimiter, skip_header=False, dtype=str)
    metadata = np.genfromtxt(metadata_filepath, delimiter=delimiter, skip_header=True, dtype=str)

    # Filter out unused IDs
    building_ids = []
    for id in data[:, 0]:
        if id not in building_ids:
            building_ids.append(id)
    bool_mask = (np.isin(metadata[:, 0], building_ids))
    metadata = metadata[bool_mask]

    # Fill the day_off column, optimized for data sorted by building ID
    day_off_column = np.zeros(shape=(data.shape[0], 1), dtype=bool) # Empty column of type bool
    current_id = "0"
    current_schedule = np.zeros(shape=(7), dtype=bool)
    for i in range(len(data)):
        if data[i][0] != current_id:
            current_id = data[i][0]
            for j in range(len(metadata)):
                if metadata[j][0] == current_id:
                    current_schedule = metadata[j][4:]
        
        current_date_str = data[i][2]
        current_date = date(int(current_date_str[:4]), int(current_date_str[5:7]), int(current_date_str[8:10]))
        day_off_column[i][0] = (current_schedule[current_date.weekday()] == "True")
    
    data_plus_column = np.concatenate([data, day_off_column], axis=1)
    np.savetxt(out_filepath, data_plus_column, delimiter=delimiter, fmt="%s")
    print(f"Added a day_off column to {in_filepath} based on {metadata_filepath} and saved to {out_filepath}")


def add_holiday_days_off(in_filepath: str, holiday_filepath: str, out_filepath: str, delimiter: str = ";") -> None:
    """Modifies the day_off column in a merged dataset to turn some of its values to True based on whether the day is a holiday or not"""
    data = np.genfromtxt(in_filepath, delimiter=delimiter, skip_header=True, dtype=str)
    holidays = np.genfromtxt(holiday_filepath, delimiter=delimiter, skip_header=True, dtype=str)
    
    # Filter out unused IDs (and sort by building ID)
    building_ids = []
    for id in data[:, 0]:
        if id not in building_ids:
            building_ids.append(id)
    bool_mask = (np.isin(holidays[:, 2], building_ids))
    holidays = holidays[bool_mask]
    holidays = holidays[np.lexsort((holidays[:, 0], holidays[:, 2]))]

    print(holidays[:30, :])

    data_size = len(data)
    holidays_size = len(holidays)

    current_building_id = -1
    i_for_same_id = -1
    print(f"0/{holidays_size}")
    for ih in range(holidays_size):
        hrow = holidays[ih]
        if current_building_id != hrow[2] or i_for_same_id == -1: # If the last holiday's building ID was NOT the same as this one's (for optimization)
            current_building_id = hrow[2]
            for i in range(data_size):  
                if data[i, 0] == current_building_id:
                    i_for_same_id = i
                    j = i
                    while j < data_size and data[j, 0] == current_building_id:
                        if data[j, 2][:10] == hrow[0]: # Compare date strings
                            data[j, 5] = True
                        j += 1
                    break
        else: # i_for_same_id stores the i at which the first row with the correct building ID was found in the last holiday
            j = i_for_same_id
            while j < data_size and data[j, 0] == current_building_id:
                if data[j, 2][:10] == hrow[0]:
                    data[j, 5] = True
                j += 1

        sys.stdout.write(f"\r{ih+1}/{holidays_size}")
    
    np.savetxt(out_filepath, data, delimiter=delimiter, fmt="%s")
    print(f"\nModified the day_off column based on {holiday_filepath} and saved to {out_filepath}")


def unwind_time_data(in_filepath: str, out_filepath: str, delimiter: str = ";") -> None:
    """
    Unwinds the timestamp column into 7 columns: year, month_sin, month_cos, weekday_sin, weekday_cos, time_of_day_sin, time_of_day_cos.
    The sin/cos fields are cyclic features, encoded to remove the ambiguity created by jumps e.g. from 24 to 1 (in the case of hours)
    time_of_day is measured in minutes since midnight of a given day.
    """
    data = np.genfromtxt(in_filepath, delimiter=delimiter, skip_header=True, dtype=str)
    timestamps = data[:, 2].astype("datetime64[m]")

    year = timestamps.astype("datetime64[Y]").astype(int) + 1970
    month = timestamps.astype("datetime64[M]").astype(int) % 12
    weekday = (timestamps.astype("datetime64[D]").astype("datetime64[W]").astype(int) + 4) % 7
    minutes = (timestamps - timestamps.astype("datetime64[D]")).astype(int)

    # Cyclic encodings
    month_sin = np.round(np.sin(2 * np.pi * month / 12), 4)
    month_cos = np.round(np.cos(2 * np.pi * month / 12), 4)
    weekday_sin = np.round(np.sin(2 * np.pi * weekday / 7), 4)
    weekday_cos = np.round(np.cos(2 * np.pi * weekday / 7), 4)
    time_sin = np.round(np.sin(2 * np.pi * minutes / (24 * 60)), 4)
    time_cos = np.round(np.cos(2 * np.pi * minutes / (24 * 60)), 4)

    time_features = np.column_stack([month_sin, month_cos, weekday_sin, weekday_cos, time_sin, time_cos]).astype(str)
    time_features = np.column_stack([year, time_features])
    result = np.column_stack([data[:, :2], time_features, data[:, 3:]])

    header = "building_id;energy;year;month_sin;month_cos;weekday_sin;weekday_cos;time_of_day_sin;time_of_day_cos;temperature;distance_to_station;day_off"
    np.savetxt(out_filepath, result, delimiter=delimiter, header=header, fmt="%s")
    print(f"Unwinded the Timestamp column from {in_filepath} and saved to {out_filepath}")

    
def extract_static_features(metadata_filepath: str, data_filepath: str, out_filepath: str, delimiter: str = ";") -> None:
    """Extracts static features from the metadata file. Currently only building surface area. Also sorts by building ID."""
    data = np.genfromtxt(data_filepath, delimiter=delimiter, skip_header=True, dtype=str)
    metadata = np.genfromtxt(metadata_filepath, delimiter=delimiter, skip_header=True, dtype=str)
    metadata = metadata[:, :2]

    # Filter out unused IDs (and sort by building ID)
    building_ids = []
    for id in data[:, 0]:
        if id not in building_ids:
            building_ids.append(id)
    bool_mask = (np.isin(metadata[:, 0], building_ids))
    metadata = metadata[bool_mask]
    metadata = metadata[metadata[:, 0].astype(int).argsort()]
    
    np.savetxt(out_filepath, metadata, delimiter=delimiter, fmt="%s")
    print(f"Extracted the static features from {metadata_filepath} and saved to {out_filepath}")


def split_dataset_train_test(train_from_id: int, in_filepath: str, out_train_filepath: str, out_test_filepath: str, delimiter: str = ";") -> None:
    """Splits the dataset into train and test sets based on train_from_id - all entries before a given building ID are extracted to the test set, rest to train set"""
    data = np.genfromtxt(in_filepath, delimiter=delimiter, skip_header=True, dtype=str)
    ids = data[:, 0].astype(int)
    split_id = ids.searchsorted(train_from_id, side="left")
    train_set = data[split_id:, :]
    test_set = data[:split_id, :]
    np.savetxt(out_train_filepath, train_set, delimiter=delimiter, fmt="%s")
    np.savetxt(out_test_filepath, test_set, delimiter=delimiter, fmt="%s")



def get_ids_in_intervals(in_filepath: str, interval: int, delimiter: str = ";") -> None:
    """Prints the unique building IDs throughout given intervals in a dataset file. Intended to ease manual train-test splitting. Expects a time-sorted file."""
    data = np.genfromtxt(in_filepath, delimiter=delimiter, skip_header=True, dtype=str)
    data_size = len(data)
    unique_ids_all = []

    i = 0
    unique_ids = []
    while i < data_size:
        curr_id = int(data[i][0])
        if curr_id not in unique_ids:
            unique_ids.append(curr_id)
        if i % interval == 0 and i!= 0:
            unique_ids_all.append(sorted(unique_ids))
            unique_ids = []
        i += 1
    
    for i in range(data_size // interval):
        print(f"{i * interval} - {(i+1) * interval}: {unique_ids_all[i]}")
    print(f"{(data_size // interval) * interval} - {data_size}: {unique_ids_all[-1]}")



def split_dataset_train_test_temporally_exclusive(test_from_point: int, test_from_id: int, in_filepath: str, out_train_filepath: str, out_test_filepath: str, delimiter: str = ";") -> None:
    data = np.genfromtxt(in_filepath, delimiter=delimiter, skip_header=True, dtype=str)
    train_set = data[:test_from_point, :]
    test_set = data[test_from_point:, :]

    train_bool_mask = (train_set[:, 0].astype(int) < test_from_id)
    train_set = train_set[train_bool_mask].astype(str)
    test_bool_mask = (test_set[:, 0].astype(int) >= test_from_id)
    test_set = test_set[test_bool_mask].astype(str)
    
    np.savetxt(out_train_filepath, train_set, delimiter=delimiter, fmt="%s")
    np.savetxt(out_test_filepath, test_set, delimiter=delimiter, fmt="%s")





def main() -> None:
    # strip_dataset(57, "./raw_data/test-data.csv", "./intermediary_data/test-stripped.csv")
    # strip_dataset(57, "./raw_data/training-data.csv", "./intermediary_data/train-stripped.csv")
    # concat_dataset("./intermediary_data/test-stripped.csv", "./intermediary_data/train-stripped.csv", "./intermediary_data/all_stripped.csv")
    # remove_redundant_columns("./intermediary_data/all_stripped.csv", [0, 3], "./intermediary_data/all_sliced.csv")
    # merge_energy_weather("./intermediary_data/all_sliced.csv", "./raw_data/weather.csv", "./intermediary_data/all_merged.csv")
    # add_weekday_days_off("./intermediary_data/all_merged.csv", "./raw_data/metadata.csv", "./intermediary_data/merged_with_day_off.csv")
    # add_holiday_days_off("./intermediary_data/merged_with_day_off.csv", "./raw_data/holidays.csv", "./intermediary_data/merged_with_day_off_holidays.csv")
    # unwind_time_data("./intermediary_data/merged_with_day_off_holidays.csv", "./processed_data/full_dataset.csv")
    # split_dataset_train_test(9, "./processed_data/full_dataset.csv", "./processed_data/train_dataset.csv", "./processed_data/test_dataset.csv") # split at about 70/30 ratio

    # extract_static_features("./raw_data/metadata.csv", "./intermediary_data/all_merged.csv", "./processed_data/static_features.csv")

    # weather = np.genfromtxt("./raw_data/weather.csv", delimiter=";", skip_header=True, dtype=str)
    # data = np.genfromtxt("./intermediary_data/all_sliced.csv", delimiter=";", skip_header=False, dtype=str)

    # data = np.genfromtxt("./intermediary_data/merged_with_day_off_holidays.csv", delimiter=";", skip_header=True, dtype=str)
    # data = data[data[:, 2].argsort()]
    # np.savetxt("./intermediary_data/timesort_merged_with_day_off_holidays.csv", data, delimiter=";", fmt="%s")

    # get_ids_in_intervals("./intermediary_data/timesort_merged_with_day_off_holidays.csv", 50000)
    # unwind_time_data("./intermediary_data/timesort_merged_with_day_off_holidays.csv", "./processed_data/alt_full_dataset.csv")
    # split_dataset_train_test_temporally_exclusive(900000, 38, "./processed_data/alt_full_dataset.csv", "./processed_data/alt_train_dataset.csv", "./processed_data/alt_test_dataset.csv")

    split_dataset_train_test(9, "./intermediary_data/merged_with_day_off_holidays.csv", "./processed_data/train_dataset_timestamped.csv", "./processed_data/test_dataset_timestamped.csv") # split at about 70/30 ratio



if __name__ == "__main__":
    main()