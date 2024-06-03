import os
import json

output_dir = "cached_result/open_natural_question/nq-test-llama-13b-sample-20"
world_rank = 3

def collect_saved_files(output_dir, world_size):
    output_dir = os.path.normpath(output_dir)
    cached_dir = output_dir.replace("$rank$", "")
    existed_file = list(os.listdir(cached_dir))


    for file in os.listdir(output_dir.replace("$rank$", "0")):
        result = {}
        part_results = []
        for i in range(world_size): 
            assert file in list(os.listdir(output_dir.replace("$rank$", str(i)))), f"can not find file '{file}' in path '{output_dir.replace('$rank$', str(i))}'"      
        if file in existed_file:
            continue
        for i in range(world_size): 
            rank_path = output_dir.replace("$rank$", str(i))       
            part_result = json.load(open(os.path.join(rank_path, file), "r", encoding="UTF-8"))


            if ("active_methods" in part_result) or ("config" in part_result):
                if i == 0:
                    result = part_result
                else:
                    if "processed_knowledge" in result:
                        result["processed_knowledge"].update(part_result["processed_knowledge"])
                    elif "scores" in result:
                        result["scores"].update(part_result["scores"])
                    else:
                        raise AttributeError("Something Wrong happened")

            else:
                
                
                result.update(part_result) 
               
        
        json.dump(result, open(os.path.join(cached_dir, file) , "w", encoding="UTF-8"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    path = os.path.join(os.path.abspath(output_dir), "$rank$")
    collect_saved_files(path, world_rank)