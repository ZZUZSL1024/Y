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
config["fragment_queue_name"] = os.environ.get(
    "FRAGMENT_QUEUE_NAME", config.get("fragment_queue_name", "")
)
config["fragment_exchange"] = os.environ.get(
    "FRAGMENT_EXCHANGE", config.get("fragment_exchange", "")
)
config["fragment_routing_key"] = os.environ.get(
    "FRAGMENT_ROUTING_KEY", config.get("fragment_routing_key", "")
)
config["fragment_result_exchange"] = os.environ.get(
    "FRAGMENT_RESULT_EXCHANGE", config.get("fragment_result_exchange", "")
)
config["fragment_result_routing_key"] = os.environ.get(
    "FRAGMENT_RESULT_ROUTING_KEY", config.get("fragment_result_routing_key", "")
)
config["fragment_index"] = os.environ.get(
    "FRAGMENT_INDEX", config.get("fragment_index", "")
)
config["fragment_user_field"] = os.environ.get(
    "FRAGMENT_USER_FIELD", config.get("fragment_user_field", "")
)
config["fragment_multimodal_field"] = os.environ.get(
    "FRAGMENT_MULTIMODAL_FIELD", config.get("fragment_multimodal_field", "")
)
config["fragment_id_field"] = os.environ.get(
    "FRAGMENT_ID_FIELD", config.get("fragment_id_field", "")
)
config["multimodal_glm_model"] = os.environ.get(
    "MULTIMODAL_GLM_MODEL", config.get("multimodal_glm_model", "")
)
config["multimodal_glm_timeout"] = os.environ.get(
    "MULTIMODAL_GLM_TIMEOUT", config.get("multimodal_glm_timeout", "")
)
