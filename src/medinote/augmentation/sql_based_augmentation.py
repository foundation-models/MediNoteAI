from pandas import DataFrame, read_parquet
from medinote import (
    dynamic_load_function_from_env_varaibale_or_config,
    initialize,
    merge_parquet_files,
)
from medinote.cached import write_dataframe


config, logger = initialize()


""" 
develoging based on this plan: 
https://chat.openai.com/share/b5cc5846-141a-4b57-8560-8065236552d8
"""
get_fields_from_obj_name_function = dynamic_load_function_from_env_varaibale_or_config(
    "get_fields_from_obj_name_function"
)


def generate_sql_schema(obj_name, fields):
    """
    chatGpt developed the initia code https://chat.openai.com/share/b9ad9d91-7b8e-4bbd-ad83-de05ba481813
    """

    # Define the mapping from Python types to SQL types
    type_mapping = {
        int: "INTEGER",
        float: "FLOAT",
        bool: "BOOLEAN",
        # Add other type mappings here
    }

    # Start building the SQL schema
    sql_schema = f"CREATE TABLE {obj_name} (\n"

    # Process each field
    for field_name, field_type in fields:
        sql_type = type_mapping.get(field_type, "VARCHAR")  # Default to VARCHAR
        sql_schema += f"    {field_name} {sql_type},\n"

    # Remove the last comma and close the statement
    sql_schema = sql_schema.rstrip(",\n") + "\n);"

    return sql_schema


# Example usage
fields = [("id", int), ("name", str), ("balance", float)]


def develop_sql_schema(obj_name: str = None):
    """_summary_

    Develop a SQL schema, selectively incorporating fields while excluding those of the Lookup type.

    Args:
        main_object (str, optional): _description_. Defaults to None.
    """
    obj_name, fields = get_fields_from_obj_name_function(obj_name)
    sql_schema = generate_sql_schema(obj_name, fields)
    return sql_schema


def generate_sql_schema_from_df(
    df: DataFrame = None, obj_name_column: str = None, persist: bool = True
):
    """_summary_

    Generate a SQL schema from a DataFrame.

    Args:
        df (DataFrame, optional): _description_. Defaults to None.
        obj_name_column (str, optional): _description_. Defaults to None.
    """
    if df is None:
        input_path = config.sql_schema.get("input_path")
        if not input_path:
            raise ValueError(f"No input_path found.")
        logger.debug(f"Reading the input parquet file from {input_path}")
        df = read_parquet(input_path)
        # df = df[:100]

    obj_name_column = obj_name_column or config.sql_schema.get("obj_name_column")

    # Apply embed_row function to each row of the DataFrame in parallel.
    # The result is a Series with lists of sql_schemas.
    logger.debug(f"Applying the sql_schema function to the DataFrame")

    chunk_size = 1000
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunk = df[i * chunk_size : (i + 1) * chunk_size]
        chunk = chunk.apply(
            lambda row: develop_sql_schema(row, column_name=obj_name_column), axis=1
        )
        df[i * chunk_size : (i + 1) * chunk_size] = chunk

    output_path = config.sql_schema.get("output_path")
    if persist and output_path:
        logger.debug(f"Writing the output parquet file to {output_path}")
        df.to_parquet(output_path, index=False)

    return df


def merge_all_sql_schema_files(pattern: str = None, output_path: str = None):
    pattern = pattern or config.sql_schema.get("merge_pattern")
    output_path = output_path or config.sql_schema.get("merge_output_path")
    df = merge_parquet_files(pattern)
    write_dataframe(df=df, output_path=output_path)


if __name__ == "__main__":
    develop_sql_schema("company")
    # generate_sql_schema_from_df()
    # merge_parquet_files()
    pass
