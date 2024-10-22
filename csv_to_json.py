import csv
import json
import pandas as pd
import os
import re


JSON_PATH = 'test_data_2/vps/vp_true'
data = pd.read_csv('test_data_2/vps/vp_true_csv/vps_51.csv')
# print(data)
def csv_to_json(csv_file_path):
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]

    # Use regex to find the first sequence of digits in the file name
    match = re.search(r'\d+', base_name)

    if match:
        number = match.group()  # Get the number as a string
    else:
        raise ValueError("No number found in the CSV file name.")

    # Create the JSON file name as 'vanishing_point_<number>_true.json'
    json_file_path = f"{JSON_PATH}/vanishing_point_{number}_true.json"
    df = pd.read_csv(csv_file_path, index_col=False)

    df = df[['X', 'Y']]
    df = df[abs(df['Y']) < 480]

    df_melted = pd.melt(df, var_name='variable', value_name='vanishing_points')

    df_melted['counter'] = df_melted.groupby('variable').cumcount() + 1
    df_melted['variable'] = df_melted['variable'].str.lower() + df_melted['counter'].astype(str)

    df_melted = df_melted.sort_values(by='counter').drop('counter', axis=1).reset_index(drop=True)

    vanishing_points_dict = df_melted.set_index('variable')['vanishing_points'].to_dict()

    data = {
        "vanishing_points": vanishing_points_dict
    }
    print(json_file_path)
    # Open the JSON file for writing
    with open(json_file_path, 'w') as json_file:
        # Write the data dictionary to the JSON
        json.dump(data, json_file, indent=4)


# Example usage:

csv_path = 'test_data_2/vps/vp_true_csv'
# csv_path = '../test_data_2/vps/vp_true_csv/vps_27.csv'
# json_path = 'test_data_2/vps/vp_true/data.json'

# csv_to_json(csv_path)


for path in os.listdir(csv_path):
    csv_path = f'test_data_2/vps/vp_true_csv/{path}'
    csv_to_json(csv_path)


