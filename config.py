import os
import tomllib

# Default to config.toml in the current directory unless CONFIG_PATH is provided
CONFIG_PATH = os.environ.get(
    "CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.toml")
)

with open(CONFIG_PATH, "rb") as f:
    config = tomllib.load(f)

# API key is retrieved from the environment for security
config["glm_api_key"] = os.environ.get("GLM_API_KEY", "")
