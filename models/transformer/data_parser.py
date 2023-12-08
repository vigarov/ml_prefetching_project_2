import json5
import torch 
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
            data[i] = {"address": hex(data[i]["address"])}
    return data

def label(data, source_window: int, target_window: int): 
    data_source=  []
    #collect windows 
    for i in range(len(data)-source_window-target_window):
        datum = {}
        x = ""
        for j in range(i, i+source_window):
            x = x + " " + (data[j]["address"])
        y = ""
        for j in range(i+source_window, i+source_window+target_window):
            y = y + " " + (data[j]["address"])
        datum["prev_faults"] = x
        datum["next_faults"] = y
        data_source.append(datum)

    return data_source


def load_pandas(filename):
    return pd.read_csv(filename, index_col=0).to_dict('records')

if __name__ == "__main__":
    data= load_json("/home/alex/ml_prefetching/dataset_gathering/correct_out_3.txt")
    ds = label(data, 10, 10)
    print(ds[:2])
    print(data[0:40])