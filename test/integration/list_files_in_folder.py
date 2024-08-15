import os
import csv

def list_files_in_folder(folder_path: str = None, output_csv: str = None):
    folder_path = folder_path = "."
    output_csv = output_csv or "output_files_list.csv"
    # Create a list to hold file details
    file_list = []

    # Recursively traverse the folder and get file names and paths
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append([file, file_path])

    # Write the list to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "File Path"])
        writer.writerows(file_list)
    return output_csv


list_files_in_folder()

