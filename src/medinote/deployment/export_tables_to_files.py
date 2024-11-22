
import os
from medinote import initialize
from medinote.embedding.vector_search import export_table_to_file

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

def export_tables_to_files(config: dict = None):
    config = config or main_config.get(export_tables_to_files.__name__)
    
    output_path = config.get("output_path")
    tables_names = config.get("table_names")

    for table_name in tables_names:
        export_table_to_file(
            config=config.get("config", {}).get(table_name, {}),
            table_name=table_name, 
            output_path=f"{output_path}/{table_name}.{config.get("type")}",
        )

if __name__ == "__main__":
    export_tables_to_files()