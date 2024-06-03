from datasets import load_dataset


def load_example(path):
    extension = path.split(".")[-1]
    if extension == "txt":
        extension = "text"
    elif extension == "jsonl":
        extension = "json"

    return load_dataset(extension, data_files=path)["train"]
