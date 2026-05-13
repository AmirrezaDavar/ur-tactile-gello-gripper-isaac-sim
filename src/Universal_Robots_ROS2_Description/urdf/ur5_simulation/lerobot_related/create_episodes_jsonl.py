import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import json

mypath = os.environ["HOME"] + '/training_data/lerobot/my_pusht/data/chunk-000'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()

jsonl_data = []

for file in onlyfiles:
    df = pd.read_parquet(mypath + '/' + file)
    print(f"file:{file} first index:{df['index'][0]}, last index:{df['index'][len(df)-1]}")
    episode_dic = {}
    episode_dic['episode_index'] = int(df['episode_index'][0])
    episode_dic['tasks'] = ["Push the T-shaped block onto the T-shaped target."]
    episode_dic['length'] = len(df)
    jsonl_data.append(episode_dic)

with open('episodes.jsonl', 'w') as f:
    for l in jsonl_data:
        f.writelines([json.dumps(l)])
        f.writelines("\n")
    