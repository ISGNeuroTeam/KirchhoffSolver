import yaml
from ot_simple_connector.connector import Connector


def get_conf(file_path="config.yaml"):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_rest_connector(file_path="config.yaml"):
    conf = get_conf(file_path)
    conf_rest = {
        k: v
        for k, v in conf["rest"].items()
        if k in ["host", "port", "user", "password"]
    }
    return Connector(**conf_rest)


def get_data_folder(file_path="config.yaml"):
    conf = get_conf(file_path)
    return conf["data"]["path"]
