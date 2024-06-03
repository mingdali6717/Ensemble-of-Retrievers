
import jsonlines
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--data_path', type=str, required=True, help='path of data to processed')
parser.add_argument('-o', '--output_path', type=str, required=True, help='path to save the processed data')

args, _ = parser.parse_known_args()

data = []
with jsonlines.open(args.data_path, mode='r') as reader:

    for line in reader:
        data.append(line)

processed_data = [{"query":q["question"], "truthful answer": q["answer"], "language": "en"} for q in data]

with jsonlines.open(args.output_path, mode="w") as writer:
    writer.write_all(processed_data)




    

