import json5
from pandas import DataFrame,read_csv
from pathlib import Path
import ast
import re

PAGE_SIZE = 4096


def get_page_address(addr: int)->int:
    return addr & ~(PAGE_SIZE-1)


def load_json(filename): 
    print("Loading data")
    # FIXME: I need bleach for my eyes Alex
    with open(filename, 'r') as file: 
        data = file.read() 
        data = data.replace("\n", " ") 
    with open(filename, 'w') as file: 
        file.write(data) 
    with open(filename, 'r') as f:
        data = json5.load(f)['a']
    print("Loaded data")
    return data

# change depending on which machine this is ran on
RAW_DATA_PATH = "/home/vigarov/ml_prefetching_project_2/data/raw/correct_out_3.txt" 
PREPROCESS_SAVE_PATH = "/home/vigarov/ml_prefetching_project_2/data/prepro/" 

PREPRO_VERSION = 1.1
PRO_VERSION = 1.5


def preprocess(path=RAW_DATA_PATH,save=True):
    assert "raw" in Path(path).absolute().as_posix()
    df = DataFrame.from_dict(load_json(RAW_DATA_PATH))
    BENCH_NAME = "canneal"
    if save:
        df.to_csv(PREPROCESS_SAVE_PATH+BENCH_NAME+"_v"+str(PREPRO_VERSION)+".csv",index=True)
    else:
        return df


def process(preprocessed_file,source_window,pred_window,page_mask,save = False):
    p = Path(preprocessed_file)
    assert p.exists() and p.is_file() and p.suffix == ".csv"
    if "prepro" in p.absolute().as_posix():
        df = read_csv(preprocessed_file)
    else:
        df = preprocess(preprocessed_file,save=False)
    y = []

    if page_mask :
        df["address"].apply(lambda address: get_page_address(int(address)))

    # Build history
    running_past_window = [hex(a) for a in df["address"][:source_window]]
    running_future_window = [hex(a) for a in df["address"][source_window:source_window+pred_window]]
    for i in range(source_window, len(df)-pred_window):
        fault_address = hex(df["address"][i])
        df["address"][i] = " ".join(running_past_window.copy())
        n_th_next_fault = hex(df["address"][i+pred_window])
        y.append(" ".join(running_future_window.copy()))
        running_past_window.pop(0)
        running_past_window.append(fault_address)
        running_future_window.pop(0)
        running_future_window.append(n_th_next_fault)
    
    # Drop unused/useless entries = first "source window" ones which don't have enough history, and last "pred_window" ones, which have no predictions
    to_drop = list(range(source_window))+list(range(len(df)-pred_window,len(df)))
    df = df.drop(to_drop) 

    assert len(df) == len(y) # Sanity check
    df["y"] = y # add output to our data frame == have one file (input+output) per trace

    df["ip"] = df["ip"].apply(lambda ip_int: hex(ip_int)) 
    df["ustack"] = df["ustack"].apply(lambda ustack_str: ustack_str.replace('"','').strip())
    df["regs"] = df["regs"].apply(lambda regs_array: ' '.join(["0x"+hex(reg)[2:].zfill(2) for reg in ast.literal_eval(regs_array.replace('"',''))])) #have regs in hex, makes more sense and reduces input size/dimension post tokenization
    df["flags"] = df["flags"].apply(lambda flag: format(flag,"016b")) # flags are bitmaps,  might be easier to interpret if we spell them out as such

    # Drop first column = index added by preprocessor
    df = df.drop(columns=df.columns[[0]])
    # Rename to reflect our new data
    df = df.rename(columns={"address":"prev_faults"})

    if save :
        df.to_csv("/home/vigarov/ml_prefetching_project_2/data/processed/processed_"+re.sub(r"v\d+\.\d+",f"v{str(PRO_VERSION)}",p.name),index=False)
    else:
        return df


if __name__ == "__main__":
    process("/home/vigarov/ml_prefetching_project_2/data/prepro/canneal_v1.1.csv",10,10,True,save=True)