# This script runs the leap prefetcher on 
# a trace. 

from pandas import *

from models.transformer.config import get_config
from models.transformer.infer import Inferer
from models.transformer.train import get_tokenizers

trace_file = read_csv("data/processed/processed_canneal_v1.4.csv")

# Function to find majority element
def findMajority(arr, n):
    candidate = -1
    votes = 0

    # Finding majority candidate
    for i in range (n):
        if (votes == 0):
            candidate = arr[i]
            votes = 1
        else:
            if (arr[i] == candidate):
                votes += 1
            else:
                votes -= 1
    count = 0

    # Checking if majority candidate occurs more than n/2
    # times
    for i in range (n):
        if (arr[i] == candidate):
            count += 1
    
    if (count > n // 2):
        return candidate
    else:
        return -1


count = 0
predictedLeap = 0
not_predictedLeap = 0
predictedOurs = 0
not_predictedOurs = 0
config = get_config()
indices_train = list(range(len(trace_file)))
src_tokenizer, tgt_tokenizer = get_tokenizers(config)
inferer = Inferer(config)
while count + 32 < len(trace_file):
    deltaLeap = trace_file['prev_faults'][count: count + 32].tolist()
    candidateOurs = inferer.infer({'prev_faults': trace_file['prev_faults'][0][count]}, 32)
    candidateLeap = findMajority(deltaLeap, 32)
    deltaLeap = trace_file[count+32]['y']
    trueValue = trace_file[count+32]['y']
    if deltaLeap == candidateLeap:
        predictedLeap += 1
    else:
        not_predictedLeap += 1

    if candidateOurs == trueValue:
        predictedOurs += 1
    else:
        not_predictedOurs += 1
    count += 1

print("Predicted by Leap: ", predictedLeap)
print("Not predicted by Leap: ", not_predictedLeap)
print("Predicted by Ours: ", predictedOurs)
print("Not predicted by Ours: ", not_predictedOurs)
