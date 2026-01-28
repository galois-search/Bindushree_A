import gc
gc.collect()
import json
import os
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Define m value
m = 12

# Function to read the results from the .txt file and return them as a list of dictionaries
def read_results_from_file(file_path):
    results = []

    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Parse each line as a JSON object and append to the results list
                result = json.loads(line.strip())  # Remove any extra whitespace
                results.append(result)
            except json.JSONDecodeError:
                # Handle the case where a line is not valid JSON (if any)
                print(f"Warning: Could not decode line: {line}")

    return results


# Function to read results from an .xlsx file
def read_results_from_excel(file_path):
    results = []

    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Convert the DataFrame into a list of dictionaries
    for index, row in df.iterrows():
        result = row.to_dict()
        results.append(result)

    return results


# Function to group results by period (length)
def group_results_by_period(results):
    period_groups = defaultdict(
        list)  # Create a dictionary where the key is the period and the value is a list of results

    for result in results:
        period_groups[result['period']].append(result)
    disp_timestamp = datetime.now()
    print(f"⏳ Completed reading one file now {disp_timestamp.strftime('%I:%M %p on %d/%m/%Y')}")
    return period_groups


# Function to write results to a separate file based on period, sorted by 'acr_max'
def write_results_to_file_by_period(period_groups, output_folder):
    for period, results in period_groups.items():
        # Sort the results by 'acr_max' in ascending order
        sorted_results = sorted(results, key=lambda x: x['acr_max'])

        # Dynamically generate the filename using the period number
        file_name = f"m{m}_unique_period_{period}.txt"
        file_path = os.path.join(output_folder, file_name)

        with open(file_path, 'w') as file:
            for result in sorted_results:
                # Write each result as a JSON line in the corresponding file
                file.write(json.dumps(result) + "\n")
        print(f"✅ Results for period {period} saved to {file_path}")


# Function to save all results to an Excel file sorted by 'period' and 'acr_max', only unique values
def save_results_to_excel(results, output_folder):
    # Convert results into a DataFrame
    df = pd.DataFrame(results)

    # Sort the DataFrame by 'period' and 'acr_max' in ascending order
    df_sorted = df.sort_values(by=['period', 'acr_max'], ascending=[True, True])

    # Drop duplicates based on 'period' and 'acr_max' columns
    df_sorted_unique = df_sorted.drop_duplicates(subset=['period', 'acr_max'])

    # Save the sorted and unique DataFrame to an Excel file
    output_file = f"m{m}_sorted_unique_results.xlsx"
    output_file_path = os.path.join(output_folder, output_file)
    df_sorted_unique.to_excel(output_file_path, index=False)
    print(f"✅ Sorted and unique results saved to {output_file_path}")


# Main execution
if __name__ == "__main__":
    folder_name = str(m)  # Folder name is the value of m (e.g., "7")

    # Create a new folder with name m_op
    output_folder = f"{folder_name}_op"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ Created output folder: {output_folder}")

    # Get a list of all .txt files in the folder
    txt_file_paths = [os.path.join(folder_name, file) for file in os.listdir(folder_name) if file.endswith('.txt')]
    # Get a list of all .xlsx files in the folder
    xlsx_file_paths = [os.path.join(folder_name, file) for file in os.listdir(folder_name) if file.endswith('.xlsx')]

    all_results = []

    # Read results from all .txt files in the folder
    for file_path in txt_file_paths:
        print(f"Reading results from {file_path}")
        results = read_results_from_file(file_path)
        all_results.extend(results)

    # Read results from all .xlsx files in the folder
    #for file_path in xlsx_file_paths:
    #    print(f"Reading results from {file_path}")
    #    results = read_results_from_excel(file_path)
    #    all_results.extend(results)

    # Group the results by period (length)
    period_groups = group_results_by_period(all_results)

    # Write the grouped results to separate files for each period, sorted by acr_max
    write_results_to_file_by_period(period_groups, output_folder)

    # Save all results to a single Excel file, sorted by 'period' and 'acr_max', only unique values
    save_results_to_excel(all_results, output_folder)