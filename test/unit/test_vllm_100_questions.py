import pandas as pd
from openai import OpenAI

# before testing it, run make distributed in makefile

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

prompt_instruction = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{user_question}`
{instructions}

DDL statements:
{create_table_statements}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{user_question}`:
```sql
"""
table_statements = """
CREATE TABLE Contacts (
    Contact_Id VARCHAR(255),
    First_Name VARCHAR(255),
    Last_Name VARCHAR(255),
    Middle_Name VARCHAR(255),
    Title VARCHAR(255),
    LinkedIn_URL VARCHAR(255),
    LinkedIn_Connections_Count INT,
    Last_Update_Date DATETIME,
    Country VARCHAR(50),
    Continent VARCHAR(50),
    City VARCHAR(255),
    State_Province_Abbr VARCHAR(50),
    State_Province VARCHAR(50),
    Professional_Email VARCHAR(255),
    Professional_Email_Validation_Status VARCHAR(50),
    Professional_Email_Validation_Status_Date DATETIME,
    Previous_Emails VARCHAR(255),
    Business_Phone VARCHAR(255),
    Contact_Job_Start_Date DATETIME,
    Job_Functions VARCHAR(50),
    Title_Levels VARCHAR(50),
    Company_Ownership_Status VARCHAR(50),
    Company_Local_Office_Address VARCHAR(255),
    Company_Local_Office_Address_2 VARCHAR(255),
    Company_Local_Office_City VARCHAR(255),
    Company_Local_Office_State_Province_Abbr VARCHAR(50),
    Company_Local_Office_State_Province VARCHAR(50),
    Company_Local_Office_Postal_Code VARCHAR(255),
    Company_Local_Office_Country VARCHAR(50),
    Company_Local_Office_Phones VARCHAR(255),
    Company_Industry VARCHAR(50),
    Company_SIC_4_Code VARCHAR(255),
    Company_SIC_4_Description VARCHAR(50),
    Company_Sector VARCHAR(50),
    Company_LinkedIn_Url VARCHAR(255),
    Company_LinkedIn_Followers INT,
    Company_Number_Of_Employees INT,
    Company_Number_Of_Employees_Range VARCHAR(255),
    Company_Revenue INT,
    Company_Revenue_Range VARCHAR(255),
    Company_Domain VARCHAR(255),
    Company_HQ_Id VARCHAR(255),
    Company_HQ_Name VARCHAR(255),
    Company_HQ_Address VARCHAR(255),
    Company_HQ_Address_2 VARCHAR(255),
    Company_HQ_City VARCHAR(255),
    Company_HQ_State_Province_Abbr VARCHAR(50),
    Company_HQ_State_Province VARCHAR(50),
    Company_HQ_Postal_Code VARCHAR(255),
    Company_HQ_Country VARCHAR(50),
    Company_HQ_Phone VARCHAR(255),
    Is_Primary_Contact BOOLEAN,
    EntryId INT,
    Created DATETIME,
    Modified DATETIME,
    Link_Action_Date DATETIME,
    Last_Updated_Date DATETIME,
    Link_Action_Method VARCHAR(255),
    Link_Action_By VARCHAR(255),
    Link_Action VARCHAR(255),
    Full_Name VARCHAR(255),
    Experience_Details VARCHAR(255),
    Company_NAICS_6_Code_2022 VARCHAR(255),
    Company_NAICS_6_2022_Description VARCHAR(50)
);

CREATE TABLE Companies (
    Company_Id INT,
    Name VARCHAR(255),
    Type VARCHAR(100),
    LinkedIn_URL VARCHAR(255),
    Parent_Company INT,
    Parent_Company_Id VARCHAR(255),
    Last_Update_Date DATETIME,
    SIC_2_Code VARCHAR(255),
    SIC_2_Description VARCHAR(100),
    SIC_4_Code VARCHAR(255),
    SIC_4_Description VARCHAR(100),
    NAICS_4_Code VARCHAR(255),
    NAICS_4_Description VARCHAR(100),
    NAICS_6_Code_2022 VARCHAR(255),
    NAICS_6_2022_Description VARCHAR(100),
    Industry VARCHAR(100),
    Industry_Description VARCHAR(255),
    Sector VARCHAR(100),
    Address VARCHAR(255),
    Address_2 VARCHAR(255),
    City VARCHAR(255),
    Country_Code VARCHAR(100),
    Country VARCHAR(100),
    Region_Code VARCHAR(255),
    Region VARCHAR(100),
    Full_Address VARCHAR(255),
    Phone VARCHAR(255),
    Phone_Toll_Free VARCHAR(255),
    Other_Phones VARCHAR(255),
    Fax VARCHAR(255),
    Website VARCHAR(255),
    LinkedIn_Company_Id VARCHAR(255),
    Year_Founded INT,
    Number_of_Branches INT,
    Facebook_URL VARCHAR(255),
    Similar_Companies INT,
    Postal_Code VARCHAR(255),
    Logo VARCHAR(255),
    Twitter_URL VARCHAR(255),
    Yelp_URL VARCHAR(255),
    Ownership_Status VARCHAR(100),
    Operating_Status VARCHAR(100),
    Crunchbase_URL VARCHAR(255),
    Exchange VARCHAR(100),
    Market_Cap INT,
    Ticker VARCHAR(255),
    Other_Websites VARCHAR(255),
    Specialties VARCHAR(255),
    Number_of_LinkedIn_Followers INT,
    Number_of_Employees INT,
    Number_of_Employees_Range VARCHAR(255),
    Revenue INT,
    Revenue_Range VARCHAR(255),
    Description VARCHAR(255),
    EntryId INT,
    Created DATETIME,
    Modified DATETIME,
    Link_Action_Date DATETIME,
    Last_Updated_Date DATETIME,
    Link_Action_Method VARCHAR(255),
    Link_Action_By VARCHAR(255),
    Link_Action VARCHAR(255),
    State VARCHAR(100),
    Parent_Company_Name VARCHAR(255),
    Is_Website_Validated BOOLEAN
);
"""

questions = pd.read_csv('/home/agent/workspace/query2sql2api/datasets/synthetic/100-company-questiona.txt',
                        sep=",",
                        header=None)

prompts = [[prompt_instruction.format(
            user_question=question,  # user_question
            instructions="",  # instructions
            create_table_statements=table_statements,  # create_table_statements
        )]
           for question in questions[1][:20]]

answers = []

for prompt in prompts:
    completion = client.completions.create(model="/mnt/models/llama-3-sqlcoder-8b",
                                      prompt=prompt,
                                      max_tokens=2048)
    print("Completion result:", completion)
    answers.append(completion.choices[0].text)
    
print(answers)