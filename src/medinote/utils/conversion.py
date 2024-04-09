import re
import json

def remove_order_by(text):
    return re.sub(r'(?i)order by.*', '', text)

# def convert_ilike_to_like(text):
#     return re.sub(r'(?i)ilike', 'like', text)

def replace_ilike_with_like(text):
    words = text.split()
    replaced_words = [word if word.lower() != 'ilike' else 'like' for word in words]
    return ' '.join(replaced_words)

def convert_sql_query(query):
    # Regular expression to find 'SELECT ... FROM' pattern, ignoring case
    pattern = re.compile(r'select .+? from', re.IGNORECASE)
    
    # Replace found pattern with 'SELECT * FROM'
    new_query = re.sub(pattern, 'SELECT * FROM', query)
    
    return new_query


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

def string2json(json_string):
    try:
        data = json.loads(json_string)
        return data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e.msg}")
        print(f"Error at character {e.pos}: {json_string[max(0, e.pos - 10):e.pos + 10]}")
        return None