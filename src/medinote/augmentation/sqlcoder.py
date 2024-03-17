

import os
from pandas import DataFrame, Series, concat, merge, read_csv, read_parquet
from medinote import dynamic_load_function_from_env_varaibale_or_config, initialize
from medinote.augmentation.datafarme_search import search_df
from medinote.augmentation.sql_based_augmentation import generate_sql_schema
from medinote.curation.rest_clients import generate_via_rest_client
from medinote.augmentation.sql_based_augmentation import generate_sql_schema


config, logger = initialize()

get_fields_from_obj_name_function = dynamic_load_function_from_env_varaibale_or_config(
    "get_fields_from_obj_name_function")
list_obj_names_function = dynamic_load_function_from_env_varaibale_or_config(
    "list_obj_names_function")
""" 
develoging based on this plan: 
https://chat.openai.com/share/b5cc5846-141a-4b57-8560-8065236552d8
"""


def generate_schema_df(obj_name: str = None,
                       given_schema: str = None
                       ):
    """_summary_

    Generate a schema DataFrame based on a specified object name.
    """
    obj_name, fields = get_fields_from_obj_name_function(obj_name)
    schema = given_schema or generate_sql_schema(obj_name, fields)
    df = DataFrame(columns=['obj_name', 'schema', 'field', 'type'])
    for field in fields:
        new_row = {'obj_name': obj_name, 'schema': schema,
                   'field': field[0], 'type': field[1]}
        df = concat([df, DataFrame(new_row, index=[0])], ignore_index=True)
    return df


def generate_synthetic_data(row: Series,
                            obj_name: str = None,
                            schema_df: DataFrame = None
                            ):
    """_summary_

    Generate synthetic data based on a specified object name.
    """
    try:
        if schema_df is None and obj_name:
            schema_df = generate_schema_df(obj_name)
        elif obj_name is None and schema_df is None:
            raise ValueError(f"neither schema_df nor obj_name is provided")

        input_column = config.sqlcoder.get("input_column") or "text"
        question = row[input_column]

        if not obj_name:
            obj_name_column = config.sqlcoder.get(
                "obj_name_column") or "obj_name"
            obj_name = row[obj_name_column]

        sql_schema = schema_df[schema_df['obj_name'].str.lower(
        ) == obj_name]['schema'].values[0]
        row_dict = {'question': question, 'ddl': sql_schema}

        template = config.sqlcoder['prompt_template']
        logger.debug(f"Using template: {template}")
        prompt = template.format(**row_dict)

        prompt_column = config.sqlcoder['prompt_column']
        if prompt_column:
            row[prompt_column] = prompt

        template = config.sqlcoder.get('payload_template')
        payload = template.format(**{"prompt": prompt})

        inference_url = config.inference.get('inference_url')
        response = generate_via_rest_client(payload=payload,
                                            inference_url=inference_url
                                            )
        output_column = config.sqlcoder.get(
            "output_column") or "inference"

        row[output_column] = response.replace('\n', ' ').strip()

        return row
    except Exception as e:
        logger.error(f"Error generating synthetic data: {repr(e)}")
        return row


def parallel_generate_synthetic_data(obj_name: str = None,
                                     df: DataFrame = None,
                                     df_processed: DataFrame = None,
                                     given_schema: str = None
                                     ):
    """_summary_

    Generate synthetic data based on a specified object name.
    """
    # if obj_name:
    #     df = search_df(obj_name, df=df)
    if df is None:
        df = prepare_data_for_inference()

    if obj_name:
        if not given_schema:
            obj_name_column = config.sqlcoder.get("obj_name_column") or "obj_name"
            df = df[df[obj_name_column] == obj_name]
        schema_df = generate_schema_df(obj_name, given_schema)
    else:
        schema_df = None
    text_column = config.sqlcoder.get("text_column") or "text"

    processed_path = config.sqlcoder.get("processed_path")

    if df_processed is None and processed_path:
        logger.debug(
            f"Reading the processed parquet file from {processed_path}")
        df_processed = read_parquet(processed_path)

    if df_processed is not None:
        # Perform a left merge with an indicator
        merged = merge(df, df_processed[[text_column]],
                       on=text_column, how='left', indicator=True)

        # Filter rows where '_merge' is 'left_only'
        df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    output_path = config.sqlcoder.get("output_path")

    chunk_size = 10
    num_chunks = len(df) // chunk_size + 1

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        chunk_df = df[start_index:end_index]

        output_file = f"{output_path}_{obj_name}_{start_index}_{end_index}.parquet" if output_path else None
        if output_file is None or not os.path.exists(output_file):
            try:
                chunk_df = chunk_df.parallel_apply(
                    generate_synthetic_data, axis=1, schema_df=schema_df, obj_name=obj_name)
            except ValueError as e:
                if "Number of processes must be at least 1" in str(e):
                    logger.error(
                        f"No idea for error: Number of processes must be at least \n ignoring .....")
            except Exception as e:
                logger.error(f"Error generating synthetic data: {repr(e)}")

            if output_file:
                try:
                    chunk_df.to_parquet(output_file)
                except Exception as e:
                    logger.error(
                        f"Error saving the embeddings to {output_file}: {repr(e)}")
        else:
            logger.info(
                f"Skipping chunk {start_index} to {end_index} as it already exists.")


def sql_generate_for_all_objects():
    objec_names = list_obj_names_function()
    for obj_name in objec_names:
        parallel_generate_synthetic_data(obj_name)


def export_generated_schema():
    objec_names = list_obj_names_function()

    dataframes = []

    for obj_name in objec_names:
        dataframes.append(generate_schema_df(obj_name))

    df = concat(dataframes, ignore_index=True)
    df.to_parquet(config.sqlcoder.get("schema_output_path"))

    return df

def prepare_data_for_inference():
    """_summary_

    Prepare data for inference.
    """
    input_path = config.sqlcoder.get("input_path")
    if input_path:
        if input_path.endswith('.parquet'):
            df = read_parquet(input_path)
        elif input_path.endswith('.csv'):
            df = read_csv(input_path)
        elif input_path.endswith('.txt'):
            df = read_csv(input_path, sep='\t', header=None, names=['text'])
        else:
            raise ValueError(
                f"Unsupported file format for {input_path}. Only .parquet, .csv, and .txt are supported.")
        df.drop_duplicates(inplace=True)
        logger.info(f"Read {len(df)} rows from {input_path}")
        return df
    else:
        raise ValueError(f"input_path is not provided")
        
            


if __name__ == "__main__":
#     given_schema = """CREATE TABLE Asset ( AssetName VARCHAR, Address VARCHAR, SquareFeet INTEGER, AssetDescription VARCHAR, City VARCHAR, Country VARCHAR)
#                 CREATE TABLE AssetFinancial ( FOREIGN KEY (Asset) REFERENCES Asset(AssetName), AllinBasis INTEGER, AmoritizationPeriodyrs INTEGER, AnalysisPeriodYrs INTEGER, CapYear1NOIMethodPurchasePrice INTEGER)
#                 """
#     given_schema = """-- Creating the Asset table
# CREATE TABLE Asset (
#     AssetName VARCHAR(255) PRIMARY KEY,
#     AssetType VARCHAR(255),
#     AssetSubType VARCHAR(255),
#     Address VARCHAR(255),
#     SquareFeet INTEGER,
#     AssetDescription VARCHAR(255),
#     City VARCHAR(255),
#     Country VARCHAR(255),
#     Watchlist ENUM('Yes', 'No'),
#     State VARCHAR(255),
#     Strategy VARCHAR(255),
#     FOREIGN KEY (AssetType) REFERENCES RefAssetTypes(TypeName);
#     FOREIGN KEY (AssetSubType) REFERENCES RefAssetSubTypes(SubTypeName);
#     FOREIGN KEY (Strategy) REFERENCES RefStrategies(StrategyName)
# );

# -- Creating the AssetFinancial table
# CREATE TABLE AssetFinancial (
#     AssetFinancialID INT PRIMARY KEY AUTO_INCREMENT,
#     AssetName VARCHAR(255),
#     AssetOfInterest VARCHAR(255),
#     DateAdded VARCHAR(255),
#     Status VARCHAR(255),
#     Strategy VARCHAR(255),
#     FOREIGN KEY (AssetName) REFERENCES Asset(AssetName),
#     FOREIGN KEY (Status) REFERENCES RefStatuses(StatusName),
#     FOREIGN KEY (Strategy) REFERENCES RefStrategies(StrategyName)
# );

# -- Creating the RefStatuses table
# CREATE TABLE RefStatuses (
#     StatusName VARCHAR(255) PRIMARY KEY
# );

# -- Creating the RefStrategies table
# CREATE TABLE RefStrategies (
#     StrategyName VARCHAR(255) PRIMARY KEY
# );


# -- Creating the RefAssetTypes table
# CREATE TABLE RefAssetTypes (
#     TypeName VARCHAR(255) PRIMARY KEY
# );

# -- Creating the RefAssetSubTypes table
# CREATE TABLE RefAssetSubTypes (
#     SubTypeName VARCHAR(255) PRIMARY KEY,
#     TypeName VARCHAR(255),
#     FOREIGN KEY (TypeName) REFERENCES RefAssetTypes(TypeName)
# );

# -- Inserting values into RefStatuses
# INSERT INTO RefStatuses (StatusName) VALUES ('Active'), ('Close'), ('Dead'), ('On Hold'), ('Watchlist');

# -- Inserting values into RefStrategies
# INSERT INTO RefStrategies (StrategyName) VALUES ('Core'), ('Core Plus'), ('Value Add'), ('Opportunistic'), ('Debt'), ('Distressed');


# -- Inserting values into RefAssetTypes
# INSERT INTO RefAssetTypes (TypeName) VALUES ('Hospitality'), ('Office'), ('Retail'), ('Industrial'), ('Multifamily'), ('Land');

# -- Inserting values into RefAssetSubTypes
# INSERT INTO RefAssetSubTypes (SubTypeName, TypeName) VALUES 
# ('Full Service Hotel', 'Hospitality'),
# ('Limited Service Hotel', 'Hospitality'),
# ('Resort/Casino', 'Hospitality'),
# ('Low Rise', 'Office'),
# ('Mid Rise', 'Office'),
# ('Farm', 'Agriculture'),
# ('Ranch', 'Agriculture'),
# ('Warehouse', 'Industrial'),
# ('Manufacturing', 'Industrial'),
# ('Office Showroom', 'Industrial'),
# ('Flex Space', 'Industrial'),
# ('Research & Development', 'Industrial'),
# ('Hospital', 'Medical'),
# ('Outpatient', 'Medical'),
# ('Medical Office', 'Medical'),
# ('Single Family', 'Residential'),
# ('Conventional Multi Family', 'Multi-Family'),
# ('Neighborhood', 'Retail'),
# ('Community', 'Retail'),
# ('Regional', 'Retail'),
# ('Super Regional', 'Retail'),
# ('Speciality', 'Retail'),
# ('Assisted Living', 'Senior Living'),
# ('Skilled Nursing', 'Senior Living'),
# ('Garden Style', 'Student Housing'),
# ('Low-Rise', 'Student Housing'),
# ('Mid-Rise', 'Student Housing'),
# ('High-Rise', 'Student Housing'),
# ('Cottage', 'Student Housing'),
# ('Condo', 'Residential'),
# ('Land', 'Land'),
# ('Land Development', 'Land'),
# ('Garden/Low RIse', 'Multi-Family'),
# ('Mid/High Rise', 'Multi-Family')
# """

    given_schema = config.schemas.get("asset")
    parallel_generate_synthetic_data('asset', given_schema=given_schema)
