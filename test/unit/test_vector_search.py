# Generatd with CHatGPT on 2021-08-25 15:00:00 https://chat.openai.com/share/133de26b-e5f5-4af8-a990-4a2b19d02254
import logging
import os
import yaml
from medinote import DotAccessibleDict
from  medinote.embedding.vector_search import add_similar_documents, opensearch_vector_query



LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()

logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
logger.addHandler(logging.FileHandler(f"{os.path.dirname(os.path.abspath(__file__))}/../logger/{os.path.splitext(os.path.basename(__file__))[0]}.log"))

# Add logger format to show the name of file and date time before the log statement
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set up error formatter
error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line %(lineno)d')

# Apply error formatter to handlers
for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):


        handler.setFormatter(error_formatter)
# Read the configuration file
with open(f"{os.path.dirname(os.path.abspath(__file__))}/../config/config.yaml", 'r') as file:
    yaml_content = yaml.safe_load(file)

config = DotAccessibleDict(yaml_content)

def test_opensearch_vector_query():
    opensearch_vector_query(config=config, logger=logger, query="test")
    pass

def test_add_similar_documents():
    add_similar_documents(config=config, logger=logger,)



if __name__ == "__main__":
    test_add_similar_documents()