import json
import pickle
import time


def read_json(file_path):
    # Open the file and parse the JSON
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def to_unix_time(timestamp):
    return int(time.mktime(timestamp.timetuple()))
