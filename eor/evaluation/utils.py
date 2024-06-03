import json
import pandas as pd
from collections import defaultdict
from collections.abc import Iterable
import os

import json
from tabulate import tabulate
from collections import defaultdict

#NQ & TruthfulQA
ANSWER_KEYS = ["truthful answer", "query_only"]
#ELI5
#ANSWER_KEYS = ["answer", "answer name", "truthful high like answers", "no knowledge response"]

def load_datasets(path_or_data):
    """
    path_or_data: str or Dict -  the path or data itself.  data should be in the format of:
        {
        query: {
                "system_answer" : str,
                "system_answer_name": str,
                "ground_truth_answer": List[str]
                "details":{
                    "method_name_1": {
                        "knowledge": str,
                        "response": str,
                        "vote_score": float,
                        "final_score": float,
                        "reward_score": float,(optional) 
                    }
                }
        
            }
        }
    """

    if isinstance(path_or_data, str):
        with open(path_or_data, "r", encoding='utf-8') as f:
            data = json.load(f)
    elif isinstance(path_or_data, dict):
        data =  path_or_data
    
    else:
        raise ValueError(f"path or data should be one of the path stored data or tha data itself in json format")
    
    method_name = []
    for ex in data.values():
        method_name = list(ex["details"].keys())
        break


    questions = []
    knowledges = defaultdict(list)
    candidates = defaultdict(list)
    methods = defaultdict(list)
    ground_truths = []

    for q, answers in data.items():
        questions.append(q.strip())
        ground_truths.append(answers["ground_truth_answer"])
        #ground_truths.append(answers["truthful high like answers"])
        s_answer = answers["system_answer"] if answers["system_answer"] != "" else "[EMPTY_ANSWER]"
        
        candidates["system_answer"].append(s_answer)
        best_method = answers["system_answer_name"]
        methods["system_answer"].append(best_method)
        knowledges["system_answer"].append(answers["details"][best_method]["knowledge"])
        for k in method_name:
            r_answer = answers["details"][k]["response"] if answers["details"][k]["response"] != "" else "[EMPTY_ANSWER]"
            candidates[k].append(r_answer)
            knowledges[k].append(answers["details"][k]["knowledge"])
            methods[k].append(k)
    
    return questions, knowledges, candidates, ground_truths, methods

def score_merge(data1, data2):
    """
    将data2合并到data1中，输入均为ResultSaver的标准输入
    """
    # 创建一个新的数据字典来存储合并后的数据
    merged_data = {}
    # 获取字典的全部key
    model_name = list(data1.keys())
    max = len(model_name)
    keys_of_model_name = data1[model_name[max - 1]].keys()

    # 合并data1中的数据
    for model_name, model_data in data1.items():
        merged_data[model_name] = model_data

    # 合并data2中的数据
    for model_name, model_data in data2.items():
        # if model_name not in merged_data:
        #     merged_data[model_name] = model_data
        for metric_name, metric_values in model_data.items():
            #if metric_name not in ("questions", "answers", "ground_truth"):
            if metric_name not in keys_of_model_name:
                merged_data[model_name][metric_name] = metric_values

    return merged_data


class ResultSaver():
    

    info_keys = ["questions", "answers", "ground_truth", "knowledges", "methods"]

    def __init__(self, data = None, path = None, models = None, metrics = None, verbose = True):
        """
        Parameters: 
        data: Dict -  must be formatted as
        {
        "model_name_1":{
            "questions": [q1, q2, ...], 
            "answers": [a1, a2, ...], #optional
            "ground_truth": [g1,g2,...], #optional
            "metric_name_1": [s1, s2, ..],
            "metric_name_2": [...],
            ...
        },
        "model_name_2":{
        ...
        },
        ...
        }

        models: only the name given by "models" will be considered, if None, all models in the result data will be considered
        metrics: only the name given by "metrics" will be calculated, if None, all metrics will be calculated
        """
        if data is None and path is not None:
            file_extension = path.split(".")[-1].lower()
            if file_extension == "xlsx":
                data = self.from_excel(path)
            elif file_extension == "csv":
                data = self.from_csv(path)
            elif file_extension == "json":
                data = self.from_json(path)


        if models is None:
            self.model_names = list(data.keys())
        else:
            self.model_names = []
            missing_models = []
            for model_name in models:
                if model_name in data:
                    self.model_names.append(model_name)
                else:
                    missing_models.append(model_name)
                    print(f"{', '.join(missing_models)} are not given in the results data")
        
        self.questions = data[self.model_names[0]]["questions"]
        if metrics is None:
            self.metric_names = [n for n in data[self.model_names[0]].keys() if n not in self.info_keys]
        else:
            self.metric_names = []
            available_metrics = [n for n in data[self.model_names[0]].keys() if n not in self.info_keys]
            missing_metrics = []
            for metric_name in metrics:
                if metric_name in available_metrics:
                    self.metric_names.append(metric_name)
                else:
                    missing_metrics.append(metric_name)
                    print(f"{', '.join(missing_metrics)} are not given in the results data")

        self.data = self._build_df(data)
        self.scores = self.compute_score(verbose=verbose)

        
    
    def _build_df(self, data):
        new_data = defaultdict(list)

        for method, ex in data.items():
            if method not in self.model_names:
                continue

            for k,v in ex.items():
                if k not in (self.info_keys+self.metric_names):
                    continue

                new_data[k].extend(v)
                l = len(v)
            methods = [method]*l
            new_data["models"].extend(methods)
        data_df = pd.DataFrame.from_dict(new_data)
        data_df.sort_values("questions", ignore_index=True, inplace=True)
        return data_df

    def compute_score(self, return_dict=False, verbose=True, threshold=0.8):
        if "bem_score" in self.data.columns:
            metric_names = self.metric_names + ["bem_acc"]
            self.data["bem_acc"] = (self.data["bem_score"] >= threshold).astype(int)
        else:
            metric_names = self.metric_names
        
        
        
            
        result_df = self.data.groupby("models")[metric_names].mean()
        if verbose:
            print(tabulate(result_df, headers='keys', tablefmt='psql'))

        if return_dict:
            
            return result_df.to_dict()
        else:
            return result_df
    
    def add_info(self, info_mapping, info_name, key_name="questions"):
        """
        info_mapping: dict - key is the column value specified by "key_name", value is the corresponding value of "info_name"
        info_name: str 
        """
        assert key_name in self.data.columns, "key name should be one of the column names in data"
        self.data[info_name] = self.data[key_name].map(info_mapping)
    
    def __getattr__(self, attr):
        if attr in self.model_names:
            return self.data.loc[self.data['models'] == attr]
        else:
            raise AttributeError(f"class do not have attribute {attr}")

    def __getitem__(self, item):
        if item in self.questions:
            return self.data.loc[self.data["questions"] == item]
        elif item in self.model_names:
            return self.data.loc[self.data['models'] == item]
        elif isinstance(item, Iterable) and item[0] in self.questions:
            return self.data.loc[self.data["questions"].isin(item)]
        elif isinstance(item, int) or isinstance(item, slice):
            return self.data.iloc[item]
        else:
            raise IndexError(f"class do not support index {item}")
    
    def to_dict(self, score_only=False, model_first = True, records=True):
        """
        output a dictionary version of result data


        parameters:
        model_first: if True, will return {model_name: depend on "records"}, else will return {"question": {model_name}}
        records: if True, data_dict will return {model_name: [{column: value},..]}, else, will return {model_name: {columns: [value1, value2, ...]}}

        return:
        scores: dict - {metric_name: {model_name: score}}
        data_dict: dict - {model_name: {depend on "records"}}
        """
        if not score_only:
            if records:
                orient="records"
            else:
                orient="list"
            data_dict = {}
            if model_first:
                for method, group in self.data.groupby("models"):
                    data_dict[method] = group.to_dict(orient=orient)
            else:
                for question, group in self.data.groupby("questions"):
                    data_dict[question] = group.to_dict(orient=orient)

        if score_only:
            return self.scores.to_dict()
        else:
            return self.scores.to_dict(), data_dict
    
    def to_excel(self, output_dir, score_only=False, **kwargs):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"save result to {output_dir}")
        
        self.scores.to_excel(os.path.join(output_dir, "scores.xlsx"), **kwargs)
        if not score_only:
            self.data.to_excel(os.path.join(output_dir, "results.xlsx"), **kwargs)
    
    def to_csv(self, output_dir, score_only=False, **kwargs):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        
        self.scores.to_csv(os.path.join(output_dir, "scores.csv"), **kwargs)
        if not score_only:
            self.data.to_csv(os.path.join(output_dir, "results.csv"), **kwargs)
    
    def to_json(self, output_dir, score_only=False, model_first = True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.scores.to_json(os.path.join(output_dir, "scores.json"), indent=4)
        if not score_only:
            _, data_dict = self.to_dict(records=True, model_first=model_first)
            with open(os.path.join(output_dir, "results.json"), 'w') as f:
                json.dump(data_dict, f, indent=4)

    def extract_scores(self, rows):
        scores = ['f1_score', 'em_score', 'bem_score', 'questions', 'answers', 'ground_truth']
        result_dict = {score: list(rows[score]) for score in scores}
        return result_dict

    def from_excel(self, path):
        data = pd.read_excel(path)
        models = ['no knowledge response', 'answer', 'summarize google good', 'summarize google plus wiki',
                  'contrive google no truncate merged', 'gen doc']
        results = {}
        for model in models:
            model_rows = data[data["models"] == model]
            results[model] = self.extract_scores(model_rows)
        return results

    def from_csv(self, path):
        data = pd.read_csv(path)
        models = ['no knowledge response', 'answer', 'summarize google good', 'summarize google plus wiki',
                  'contrive google no truncate merged', 'gen doc']
        results = {}
        for model in models:
            model_rows = data[data["models"] == model]
            results[model] = self.extract_scores(model_rows)
        return results

    def from_json(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        result = {}
        for model, info in data.items():
            model_data = {
                "f1_score": [item["f1_score"] for item in info],
                "em_score": [item["em_score"] for item in info],
                "bem_score": [item["bem_score"] for item in info],
                "questions": [item["questions"] for item in info],
                "answers": [item["answers"] for item in info],
                "ground_truth": [item["ground_truth"] for item in info]
            }
            result[model] = model_data
        return result