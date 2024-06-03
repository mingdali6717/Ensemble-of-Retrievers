from eor.evaluation import Evaluator
import numpy as np
import pandas as pd
import argparse
import os
import debugpy

def main():
    
    parser = argparse.ArgumentParser(description='a project on retrieval enhance with controller')
    
    parser.add_argument('-r', '--result_path',type=str, required=True, default=None, help='result json file path')
    parser.add_argument('-o', '--output_path',type=str, default=None, help='path to save result')
    parser.add_argument('-n', '--dataset_name',type=str, default="NQ", help='path to save result')
    parser.add_argument("-m", "--metric", type=str, nargs="*", default=None , help='addimetrics to evaluate')
    parser.add_argument('-d', '--debug', action='store_true',help='use valid dataset to debug your system')
    
    args, _ = parser.parse_known_args()
    if args.debug:
        print("start to listen debugpy")
        debugpy.listen(("0.0.0.0", 14322))
        debugpy.wait_for_client()
# 
    

    result_path = os.path.abspath(args.result_path)
    
    if args.output_path is None:
        output_path = os.path.dirname(result_path)
    else:
        output_path = args.output_path
    print(f"load result data from '{result_path}'")

    # truthfulqa_result_path = "raw_result/v03/truthfulqa2.json"
    # tqa_output_dir = "evaluation_results/truthfulqa/trail"
    if args.metric is not None:
        metric = list(args.metric)
    else:
        metric = ["em", "bem"]
    evaluator = Evaluator(args.dataset_name, metrics= metric)
    evaluator.evaluate(result_path)
    evaluator.result.to_excel(output_path)
    

    
if __name__ =="__main__":
    main()