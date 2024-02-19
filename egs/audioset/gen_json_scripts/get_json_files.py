import csv
import json

# File path for the CSV file
csv_file_path = '/data/swarup_behera/Research/TOME/ToMe/egs/audioset/data/metadata/new_filtered_eval_segments.csv'
# Output file path for the JSON file
json_file_path = '/data/swarup_behera/Research/TOME/ToMe/egs/audioset/data/metadata/eval_segments.json'

# Initialize the data structure for the JSON file
data = {'data': []}

# Reading the CSV file and transforming the data
try:
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        count = 0
        for row in csv_reader:
            count +=1 
            #import pdb; pdb.set_trace() 
            # Extracting YTID and labels
            ytid, _, _, labels = row
            # Constructing the WAV file path
            wav_path = f"/data/swarup_behera/Research/TOME/ToMe/egs/audioset/audios/eval_segments/Y{ytid}.wav"
            #wav_path_check = f"datasets/audioset201906/audios/eval_segments/Y{ytid}.wav"
             
            # Formatting labels 
            labels = labels.strip('"')
            formatted_labels = ','.join(labels.split())
            # Removing any leading backslashes and quotes
            formatted_labels = formatted_labels.lstrip('\"\\')
            # Adding the entry to the data list
            data['data'].append({'wav': wav_path, 'labels': formatted_labels})

    print(count)
    # Writing the JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

    success = True
except Exception as e:
    success = False
    error_message = str(e)

json_file_path if success else error_message

