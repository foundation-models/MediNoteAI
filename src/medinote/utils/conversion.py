import re
import json


def is_well_formed_sql(statement):
    try:
        import sqlparse
        parsed = sqlparse.parse(statement)
        if not parsed:
            return False
        stmt = parsed[0]
        # Check if the statement starts with a SELECT
        if stmt.get_type() != 'SELECT':
            return False
        # Check for basic syntax structure # not sure about the following terms so for now let's comment them
        # if not any(token.ttype is Keyword and token.value.upper() == 'SELECT' for token in stmt.tokens):
        #     return False
        # if not any(token.ttype is Keyword and token.value.upper() == 'FROM' for token in stmt.tokens):
        #     return False
        return True
    except:
        return False
    
def extract_well_formed_sql(statements):
    # Define a regular expression pattern for potential SQL SELECT statements
    # generated with the help of GPT4 https://chat.openai.com/share/11043ac1-fbf7-4246-9d10-375248aa601f
    sql_pattern = r'\bSELECT\b.*?\bFROM\b.*?(?=(?:"|\'\'\'|\n\n|```)|$)'
    
    # Find all matches of the pattern in the input string
    potential_statements = re.findall(sql_pattern, statements, re.IGNORECASE | re.DOTALL)
    
    well_formed_statements = []
    for statement in potential_statements:
        if is_well_formed_sql(statement):
            well_formed_statements.append(statement.strip())
    
    return well_formed_statements

def replace_equals_with_like(text):
    # Define the pattern to search for xxx = "yyy"
    pattern = r'(\w+)\s*=\s*"([^"]+)"'
    
    # Replace the pattern with xxx like "%yyy%"
    result = re.sub(pattern, r'\1 like "%\2%"', text)
    
    return result

def convert_to_select_all_query(query):
    # Regular expression to find 'SELECT ... FROM' pattern, ignoring case
    pattern = re.compile(r'select .+? from', re.IGNORECASE)
    
    # Replace found pattern with 'SELECT * FROM'
    new_query = re.sub(pattern, 'SELECT * FROM ', query)
    
    return new_query

def remove_limits(sql_statement):
    # checkout https://chatgpt.com/share/decb3683-c0db-456f-b327-24237e81a0e6
    # This regex will match LIMIT followed by a number
    pattern = r'LIMIT \d+'
    # Use re.sub to replace the pattern with an empty string
    result = re.sub(pattern, '', sql_statement, flags=re.IGNORECASE)
    return result

def replace_table_name(sql_statement, new_table_name):
    # Define a regex pattern to find the word after FROM
    pattern = r'FROM\s+\w+'
    
    # Define the replacement string
    replacement = f'FROM {new_table_name}'
    
    # Replace the existing table name with the new table name
    updated_sql = re.sub(pattern, replacement, sql_statement, flags=re.IGNORECASE)
    
    return updated_sql

def has_capital_in_middle(word):
    # Check if there is a capital letter in the middle of the word
    return re.search(r'\b[A-Za-z]*[A-Z][a-z]+\b', word) is not None

def format_key(key):
    # Insert space before capital letters, skip the first character
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', key)

def add_T00_00_to_dates(text):
    """
    Search for all occurrences of dates in the 'YYYY-MM-DD' format within the input string 
    and convert them to the 'YYYY-MM-DDT00:00' format.

    Args:
    text (str): The string to be processed.

    Returns:
    str: The modified string with updated date formats.
    """
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    return re.sub(date_pattern, r'\g<0>T00:00', text)

def reformat_field_names(sql_query):
    # Extract the WHERE clause
    where_clause_match = re.search(r'WHERE (.+?)(?: ORDER BY| GROUP BY| HAVING| LIMIT|$)', sql_query, re.IGNORECASE | re.DOTALL)
    if not where_clause_match:
        return "No WHERE clause found."

    where_clause = where_clause_match.group(1)

    # Extract field names, assuming they are followed by comparison operators
    field_names = re.findall(r'\b(\w+)\s*( = | != | <> | < | > | <= | >= |=|!=|<>|<|>|<=|>=| LIKE | IN | BETWEEN | AND | OR )', where_clause, re.IGNORECASE)
    
    # Extract only unique field names
    unique_field_names = set(name for name, _ in field_names)

    fields_in_where =  list(unique_field_names)
    for field in fields_in_where:
        if has_capital_in_middle(field):
            new_format = f"[{format_key(field)}]"
            sql_query = sql_query.replace(field, new_format)
    return sql_query


def truncate_text(text, max_length=10):
    # Truncate text if it's longer than max_length
    return text if len(text) <= max_length else text[:max_length] + '...'

def string2json(json_string, default_value=None):
    try:
        data = json.loads(json_string, strict=False)
        return data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e.msg}")
        eval_string = json_string.replace('null', 'None').replace('true', 'True').replace('false', 'False')
        try:
            eval_data = eval(eval_string)
            return eval_data
        except Exception as e:
            print(f"Error at character {e.pos}: {json_string[max(0, e.pos - 10):e.pos + 10]}")
            return default_value
    except Exception as e:            
        print(f"Error at character {e.pos}: {json_string[max(0, e.pos - 10):e.pos + 10]}")
        return default_value
    
def test():
    return "HI THis is me....."

def is_dict_empty(d):
    if not d:
        return True  # The dictionary is empty
    for key, value in d.items():
        if value:  # Check if the value is not empty, None, or any falsy value
            return False
    return True  # All values are empty, None, or falsy