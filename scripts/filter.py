import os
import csv

def extract_id_from_wav(filename):
    return filename[1:-4]  # Assuming the ID starts from the second character

def filter_csv(wav_folder, csv_folder):
    # Get the list of ".wav" files
    wav_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
    #import pdb; pdb.set_trace()

    # Extract IDs from ".wav" filenames
    wav_ids = [extract_id_from_wav(f) for f in wav_files]

    # Read the ".csv" file and filter entries
    csv_path = os.path.join(csv_folder, 'actual_balanced_train_segments.csv')  # Replace with your actual CSV filename
    output_csv_path = os.path.join(csv_folder, 'filtered_balanced_train_segments.csv')  # Replace with your desired output CSV filename

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Read the header
        header = next(csv_reader)

        # Find the index of the column containing the IDs (change 'YTID' to the actual column header)
        id_column_index = header.index('# YTID')

        # Filter rows based on whether the ID is in the list of IDs from ".wav" files
        filtered_rows = [row for row in csv_reader if row[id_column_index] in wav_ids]

    # Write the filtered rows to a new CSV file
    with open(output_csv_path, 'w', newline='') as output_csv_file:
        csv_writer = csv.writer(output_csv_file)
        csv_writer.writerow(header)
        csv_writer.writerows(filtered_rows)

    print(f"Filtered CSV file written to {output_csv_path}")

# Replace the folder paths with the actual paths where your files are located

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_folder_path = os.path.join(script_dir, 'fcsv')
wav_folder_path = '/Users/karthikgowda/Downloads/audioset_tagging_cnn-master/datasets/audioset201906/audios/balanced_train_segments'

filter_csv( wav_folder_path, csv_folder_path)
