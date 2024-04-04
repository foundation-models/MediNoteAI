# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/mnt/models/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("/mnt/models/gemma-2b-it", device_map="auto")

input_text = "Write me a poem about Machine Learning."


text =  "Find all companies with a 'Cap' of $100 million and a 'DealSource' of 'Investor A'."
text =  "Find all companies with a 'Cap' of $100 million and a 'DealSource' of 'Investor A'."
text =  "Find all companies with a 'Cap' of $100 million and a 'DealSource' of 'Investor A'."
text =  "Show the companies that have participated in deals with a 'High' deal score and have a primary contact in the city of 'San Francisco'."

obj_names = "Campaign, Contact, Company, Deals, Asset"



input_text = f"Given the query, please evaluate the following list of objects:\n {obj_names}.\nRank these objects in order of their relevance to the query, starting with the most relevant.  please provide the list of top 3 objects in a comma separated list. Use only object names and no explanation.\n\nInput: {text}\n##OUTPUT\n"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_length=500, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0]))
