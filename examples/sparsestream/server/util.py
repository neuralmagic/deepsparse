import yaml

def get_tokens():

    with open("./server/config.yaml") as file: 
        config = yaml.safe_load(file.read())
    
    return config["twitter_tokens"]

