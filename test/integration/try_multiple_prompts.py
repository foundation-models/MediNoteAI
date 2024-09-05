import os
from medinote import chunk_process, read_dataframe, initialize
from pandas import DataFrame
from medinote.inference.inference_prompt_generator import row_infer
from pandas import Series

main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

config = main_config.get("try_multiple_prompts")

def request_chatgpt(text):
    row = Series({"text": text})
    row = row_infer(row=row, config=config)
    return row["answer"]

folder_path = config.get("prompts_path")

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        answer = request_chatgpt(content)
        
        new_filename = f"{os.path.splitext(filename)[0]}_ans.txt"
        new_file_path = os.path.join(folder_path, new_filename)
        with open(new_file_path, 'w', encoding='utf-8') as new_file:
            new_file.write(answer)
        
        print(f"Processed {filename}, saved answer to {new_filename}")
