import os
import jsonlines
import random
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--data_path', type=str, required=True, help='path of data to processed')
parser.add_argument('-o', '--output_path', type=str, required=True, help='path to save the processed data')
args, _ = parser.parse_known_args()

SEED = 28
NUM_TO_SAMPLE = 500


random.seed(SEED)

data = []

with jsonlines.open(args.data_path, mode='r') as reader:
    for d in reader:
        data.append(d)


sampled_data = random.sample(data, NUM_TO_SAMPLE)

with jsonlines.open(args.output_path, mode="w") as f:
    f.write_all(sampled_data)