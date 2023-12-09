import json5
from pandas import DataFrame


def load_json(filename): 
    print("Loading data")
    # FIXME: I need bleach for my eyes Alex
    #with open(filename, 'r') as file: 
    #    data = file.read() 
    #    data = data.replace("\n", " ") 
    #with open(filename, 'w') as file: 
    #    file.write(data) 
    with open(filename, 'r') as f:
        data = json5.load(f)['a']
    print("Loaded data")
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

# change depending on which machine this is ran on
RAW_DATA_PATH = "/home/vigarov/ml_prefetching_project_2/data/raw/correct_out_3.txt" 
PREPROCESS_SAVE_PATH = "/home/vigarov/ml_prefetching_project_2/data/prepro/" 

PREPRO_VERSION = 1

if __name__ == "__main__":
    pd = DataFrame.from_dict(load_json(RAW_DATA_PATH))
    BENCH_NAME = "canneal"
    pd.to_csv(PREPROCESS_SAVE_PATH+BENCH_NAME+"_v"+str(PREPRO_VERSION)+".txt")
