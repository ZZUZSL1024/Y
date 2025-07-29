import json
import os

CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.json"))

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)
