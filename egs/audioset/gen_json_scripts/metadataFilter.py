#import pandas as pd 
import re 
import os

path = "/data/swarup_behera/Research/TOME/ToMe/egs/audioset/data/metadata/eval_segments.csv"

def replace_commas_in_quotes(text):
    return re.sub(r'"([^"]*)"', lambda match: match.group(0).replace(',', ' '), text)

with open(path, 'r') as file:
    # Read the entire content of the file
    content = file.read()
    change = replace_commas_in_quotes(content)

newPath = "/data/swarup_behera/Research/TOME/ToMe/egs/audioset/data/metadata/filtered_eval_segments.csv"
with open(newPath, 'w') as file: 
    file.write(change)

    




