
from medinote.augmentation.synthetic_sql import (
    apply_template, get_fields_from_yaml_for_table,
    template_replace, augment_sql_qery, extract_table_name)


def test_get_fields_from_yaml_for_table():
    # Example Usage
    yaml_content = """
    Entity & Fields Description:
        Asset:
            Fields:
            - AssetName
            - Address
            - SquareFeet
            - AssetDescription
            - AssetSubType
            - AssetType
        AssetFinancial:
            Fields:
            - AssetofInterest
            - AllinBasis
    """

    sql_query = "select * from Asset where Address = 'xxx'"

    fields = get_fields_from_yaml_for_table(sql_query, yaml_content)
    assert fields == ['AssetName', 'Address', 'SquareFeet',
                      'AssetDescription', 'AssetSubType', 'AssetType']


def test_template_replace():
    # Test case 1: Replace single placeholder
    template = "Hello, ${name}!"
    values_dict = {"name": "John"}
    expected_result = "Hello, John!"
    assert template_replace(template, values_dict) == expected_result

    # Test case 2: Replace multiple placeholders
    template = "My name is ${name} and I am ${age} years old."
    values_dict = {"name": "Alice", "age": 25}
    expected_result = "My name is Alice and I am 25 years old."
    assert template_replace(template, values_dict) == expected_result

    # Test case 3: Placeholder not found in template
    template = "Hello, ${name}!"
    values_dict = {"age": 30}
    expected_result = "Hello, ${name}!"
    assert template_replace(template, values_dict) == expected_result

    # Test case 4: Empty template
    template = ""
    values_dict = {"name": "Bob"}
    expected_result = ""
    assert template_replace(template, values_dict) == expected_result

    # Test case 5: Empty values dictionary
    template = "Hello, ${name}!"
    values_dict = {}
    expected_result = "Hello, ${name}!"
    assert template_replace(template, values_dict) == expected_result


def test_generate_prompt_to_augment_sql():
    # Test case 1: SQL query with template provided
    sql_query = "select * from Asset where Address = 'xxx'"
    template = "Please provide values for the following fields: ${fields}"
    expected_result = "Please provide values for the following fields: AssetName, Address, SquareFeet, AssetDescription, AssetSubType, AssetType"
    assert apply_template(
        sql_query, template) == expected_result

    # Test case 2: SQL query with no template provided
    sql_query = "select * from AssetFinancial where AssetofInterest = 'yyy'"
    expected_result = "Please provide values for the following fields: AssetofInterest, AllinBasis"
    assert apply_template(sql_query) == expected_result

    # Test case 3: SQL query with empty template
    sql_query = "select * from Asset where Address = 'zzz'"
    template = ""
    expected_result = ""
    assert apply_template(
        sql_query, template) == expected_result

    # Test case 4: SQL query with no fields
    sql_query = "select * from Asset where 1 = 1"
    template = "Please provide values for the following fields: ${fields}"
    expected_result = "Please provide values for the following fields: "
    assert apply_template(
        sql_query, template) == expected_result

    # Test case 5: SQL query with special characters in fields
    sql_query = "select * from Asset where Address = 'aaa' and AssetType = 'bbb'"
    template = "Please provide values for the following fields: ${fields}"
    expected_result = "Please provide values for the following fields: AssetName, Address, SquareFeet, AssetDescription, AssetSubType, AssetType"
    assert apply_template(
        sql_query, template) == expected_result



def test_generate_prompt_to_augment_sql():
    # Test case 1: SQL query with template provided
    # sql_query = "select * from Asset where Address = 'xxx'"
    # template = "Please provide values for the following fields: ${fields}"
    # expected_result = "Please provide values for the following fields: AssetName, Address, SquareFeet, AssetDescription, AssetSubType, AssetType"
    # assert apply_template(
    #     sql_query, template) == expected_result

    # Test case 2: SQL query with no template provided
    sql_query = "select * from AssetFinancial where AssetofInterest = 'yyy'"
    expected_result = "Please provide values for the following fields: AssetofInterest, AllinBasis"
    assert apply_template(sql_query) == expected_result

    # Test case 3: SQL query with empty template
    sql_query = "select * from Asset where Address = 'zzz'"
    template = ""
    expected_result = ""
    assert apply_template(
        sql_query, template) == expected_result

    # Test case 4: SQL query with no fields
    sql_query = "select * from Asset where 1 = 1"
    template = "Please provide values for the following fields: ${fields}"
    expected_result = "Please provide values for the following fields: "
    assert apply_template(
        sql_query, template) == expected_result

    # Test case 5: SQL query with special characters in fields
    sql_query = "select * from Asset where Address = 'aaa' and AssetType = 'bbb'"
    template = "Please provide values for the following fields: ${fields}"
    expected_result = "Please provide values for the following fields: AssetName, Address, SquareFeet, AssetDescription, AssetSubType, AssetType"
    assert apply_template(
        sql_query, template) == expected_result
    

def test_augment_sql_qery():
    # Test case 1: SQL query with template provided
    # sql_query = "select * from Asset where Address = 'xxx'"
    # template = "Please provide values for the following fields: ${fields}"
    # inference_response_limit = 5
    # synthetic_queries = augment_sql_qery(sql_query, template, inference_response_limit)
    # assert len(synthetic_queries) == inference_response_limit

    # Test case 2: SQL query with no template provided
    sql_query = "select * from AssetFinancial where AssetofInterest = 'yyy'"
    inference_response_limit = 100
    synthetic_queries = augment_sql_qery(sql_query=sql_query, inference_response_limit=inference_response_limit, output_file="/tmp/output.parquet")
    assert len(synthetic_queries) == inference_response_limit

    # Test case 3: SQL query with empty template
    sql_query = "select * from Asset where Address = 'zzz'"
    template = ""
    inference_response_limit = 5
    synthetic_queries = augment_sql_qery(sql_query, template, inference_response_limit)
    assert len(synthetic_queries) == inference_response_limit

    # Test case 4: SQL query with no fields
    sql_query = "select * from Asset where 1 = 1"
    template = "Please provide values for the following fields: ${fields}"
    inference_response_limit = 5
    synthetic_queries = augment_sql_qery(sql_query, template, inference_response_limit)
    assert len(synthetic_queries) == inference_response_limit

    # Test case 5: SQL query with special characters in fields
    sql_query = "select * from Asset where Address = 'aaa' and AssetType = 'bbb'"
    template = "Please provide values for the following fields: ${fields}"
    inference_response_limit = 5
    synthetic_queries = augment_sql_qery(sql_query, template, inference_response_limit)
    assert len(synthetic_queries) == inference_response_limit
    
    
    
    
    
if __name__ == "__main__":
    # test_get_fields_from_yaml_for_table()
    # test_template_replace()
    # test_generate_prompt_to_augment_sql()
    test_augment_sql_qery()
    print("All tests passed!")