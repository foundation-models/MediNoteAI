import sys
from medinote import dynamic_load_function_from_env_varaibale_or_config, initialize

if __name__ == "__main__":
    config, logger = initialize()
    module_name = sys.argv[1]
    function = dynamic_load_function_from_env_varaibale_or_config(
        module_name, config=config
    )
    function()
