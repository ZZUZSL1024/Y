import os
import tomli

# Default to config.toml in the current directory unless CONFIG_PATH is provided
CONFIG_PATH = os.environ.get(
    "CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.toml")
)

with open(CONFIG_PATH, "rb") as f:
    config = tomli.load(f)

# Sensitive values are pulled from environment variables if available
config["glm_api_key"] = os.environ.get("GLM_API_KEY", "")
config["rabbitmq_pass"] = os.environ.get(
    "RABBITMQ_PASS", config.get("rabbitmq_pass", "")
)
config["rabbitmq_host"] = os.environ.get(
    "RABBITMQ_HOST", config.get("rabbitmq_host", "")
)
config["rabbitmq_user"] = os.environ.get(
    "RABBITMQ_USER", config.get("rabbitmq_user", "")
)
config["api_token"] = os.environ.get("API_TOKEN", config.get("api_token", ""))
config["api_base_url"] = os.environ.get(
    "API_BASE_URL", config.get("api_base_url", "")
)
config["queue_name"] = os.environ.get(
    "QUEUE_NAME", config.get("queue_name", "")
)
