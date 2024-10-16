import csv
from io import StringIO
import json
import logging
import os
from pandas import DataFrame, Series, read_csv
from yaml import safe_load
import re

from medinote.curation.rest_clients import call_openai, generate_via_rest_client


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()

logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


default_prompt2 = "Using the provided SQL template and the list of fields from the '${table_name}' table, generate ${inference_response_limit} synthetic SQL queries. The base SQL query is:\n\n${sql_query}\nThe fields in the '${table_name}' table are: ${field_names}.\n\nEach synthetic SQL query should be a variation of the base query, incorporating different combinations and conditions using the provided fields. Ensure that the queries are syntactically correct and varied in structure and complexity."
default_prompt1 = "Using the provided SQL template and the list of fields from the '${table_name}' table, generate ${inference_response_limit} synthetic SQL queries. The base SQL query is:\n\n${sql_query}\nThe fields in the '${table_name}' table are: ${field_names}.\n\nEach synthetic SQL query should be a variation of the base query, focusing on simple structures. Exclude complex queries such as joins, subqueries, and grouping. Include variations with sorting and counting. Ensure that the queries are syntactically correct and maintain a level of simplicity in structure and logic."
default_prompt3 = "Using the provided SQL template and the list of fields from the '${table_name}' table, generate ${inference_response_limit} synthetic SQL queries. The base SQL query is:\n\n${sql_query}\nThe fields in the '${table_name}' table are: ${field_names}.\n\nCreate a CSV output where each line consists of a synthetic SQL query and the corresponding used field names in array format. Focus on simple query structures, excluding complex queries such as joins, subqueries, and grouping, but include sorting and counting. Ensure that the queries are syntactically correct and maintain simplicity. The CSV format should be: 'Query, [Fields]'."
default_prompt = "Generate ${inference_response_limit} synthetic SQL queries for the '${table_name}' table with fields ${field_names}, based on the base query '${sql_query}', focusing on simple structures (no joins, subqueries, grouping; include sorting and counting), and format the output as a CSV with each line containing 'Query, [Fields]'."
default_instruction = "Generate a specified number of synthetic SQL queries based on a given table and its fields, and output them in a CSV format with each query and the corresponding used field names."


def extract_table_name(sql_query):
    """ Extract table name from the SQL query """
    match = re.search(r"from\s+(\w+)", sql_query, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None


def get_fields_from_yaml_for_table(sql_query: str, 
                                   table_fields_mapping_file: str = None, 
                                   root_key: str = 'Entity & Fields Description'
                                   ):

    with open(table_fields_mapping_file, 'r') as file:
        table_fields_mapping_file = file.read()

    """ Get fields for a given table name from the YAML data """
    # Parsing the YAML content
    yaml_data = safe_load(table_fields_mapping_file)

    # Extracting table name from SQL query
    table_name = extract_table_name(sql_query)

    entity_data = yaml_data.get(root_key, {})

    # Getting fields for the extracted table name
    fields = entity_data.get(table_name, {}).get('Fields', [])
    return fields, table_name


def template_replace(template, values_dict):
    # This function searches for placeholders in the template and replaces them with the corresponding values from the dictionary
    result = template
    for key, value in values_dict.items():
        result = result.replace(f'${{{key}}}', str(value))
    return result


def apply_query_template(input: str,
                         template: str = None,
                         inference_response_limit: int = 5,
                         instruction: str = None,
                         table_fields_mapping_file: str = None,
                         ):
    """ Generate a prompt to augment the SQL query """
    # Extracting fields from the SQL query
    field_names, table_name = get_fields_from_yaml_for_table(
        sql_query=input, table_fields_mapping_file=table_fields_mapping_file) if table_fields_mapping_file else (None, None)

    # Generating a template for the prompt
    template = template or default_prompt
    template = template_replace(template, {"field_names": ", ".join(field_names),
                                           "attributes": ", ".join(field_names),
                                           "table_name": table_name,
                                           "main_object": table_name,
                                           "input": input,
                                           "sql_query": input,
                                           "inference_response_limit": inference_response_limit,
                                           "instruction": instruction
                                           }
                                )

    return template


def write_csv(csv_string: str, output_file: str):
    # Split the string into lines
    lines = csv_string.split('\n')

    # Open a file in write mode
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write each line to the CSV
        for line in lines:
            writer.writerow(line.split(','))


def filtering_sql_queries(queries: str):
    start_index = queries.lower().find("select")
    return queries[start_index:].strip() if start_index > 0 else None


def query_llm_for_dataframe(row: Series, template: str = None,
                        input_column: str = 'input',
                        output_column: str = 'inference_result',
                        inference_response_limit: int = 5,
                        instruction: str = None,
                        use_openai: bool = False,
                        inference_url: str = None,
                        payload_template: str = None,
                        output_separator: str = '|',
                        filtering_function: callable = None,
                        table_fields_mapping_file: str = None,
                        ):
    synthetic_queries = query_llm(input=row[input_column], template=template,
                   inference_response_limit=inference_response_limit,
                   instruction=instruction,
                   use_openai=use_openai,
                   inference_url=inference_url,
                   payload_template=payload_template,
                   filtering_function=filtering_function,
                   table_fields_mapping_file=table_fields_mapping_file
                   )
    if output_separator:
        synthetic_queries = synthetic_queries.split(output_separator)
    else:
        synthetic_queries = synthetic_queries.split('\n')
    row[output_column] = synthetic_queries
    return row


def query_llm(input: str, template: str = None,
              inference_response_limit: int = 5,
              instruction: str = None,
              use_openai: bool = False,
              inference_url: str = None,
              payload_template: str = None,
              filtering_function: callable = None,
              table_fields_mapping_file: str = None,
              ):
    """ Generate synthetic SQL queries by augmenting the base query """
    # Generating synthetic SQL queries
    instruction = instruction or default_instruction

    try:
        prompt = apply_query_template(
            input=input, template=template,
            inference_response_limit=inference_response_limit,
            instruction=instruction,
            table_fields_mapping_file=table_fields_mapping_file
        )
        if payload_template:
            payload = template_replace(
                payload_template, {"prompt": prompt}) if payload_template else None
            payload = payload.replace("\n", "\\n")
            payload_json = json.loads(payload)

        # synthetic_queries = generate_via_rest_client(prompt=prompt)
        synthetic_queries = call_openai(
            prompt, instruction) if use_openai else generate_via_rest_client(
                payload=payload_json,
                inference_url=inference_url,
        )

        if filtering_function:
            synthetic_queries = filtering_function(synthetic_queries)
        return synthetic_queries
        # if synthetic_queries:
        #     data = StringIO(synthetic_queries)
        #     # Read the CSV string into a DataFrame
        #     df = read_csv(data, on_bad_lines='skip', engine='python',
        #                   header=None, names=[output_column], sep=output_separator)
        #     # print(inference_response_limit)
        #     # print(df.shape)
        #     return df
        # else:
        #     return DataFrame()
    except Exception as e:
        logger.error(f"Exception: {repr(e)}")
        raise e
