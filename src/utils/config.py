import json


def load_config(config_path: str):
    f = open(config_path)
    data = json.load(f)
    f.close()
    return data
