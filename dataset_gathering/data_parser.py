import json5
import torch 
from datasets import load_dataset
import datetime
import pandas as pd

def load_json(filename): 
    with open(filename, 'r') as file: 
        data = file.read() 
        data = data.replace("\n", " ") 
    with open(filename, 'w') as file: 
        file.write(data) 
    with open(filename, 'r') as f:
        data = json5.load(f)['a']
        for i in range(len(data)):
            data[i]["address"] = hex(data[i]["address"])
    return data

def label(data, source_window: int, target_window: int): 
    data_source=  []
    #collect windows 
    for i in range(len(data)-source_window-target_window):
        datum = {}
        x = []
        for j in range(i, i+source_window):
            x.append(data[j]["address"])
        y = []
        for j in range(i+source_window, i+source_window+target_window):
            y.append(data[j]["address"])
        datum["x"] = x
        datum["y"] = y
        data_source.append(datum)

    return data_source



if __name__ == "__main__":
    start=datetime.datetime.now()
    # data= load_json("/home/alex/ml_prefetching/dataset_gathering/correct_out_3.txt") #0:01:37.987487 to run
    data =  pd.read_csv('/home/alex/ml_prefetching/dataset_gathering/pandify.csv', index_col=0).to_dict('records')
    ds = label(data, 10, 10)
    end = datetime.datetime.now()
    print("took", end-start, "to run")
    print(ds[:2])
    print(data[0:40])