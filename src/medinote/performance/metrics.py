import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import os
from pandas import DataFrame
import yaml
from medinote import initialize, read_dataframe, write_dataframe
from medinote.finetune.dataset_generation import generate_jsonl_dataset
from rouge import Rouge 

_, logger = initialize()

with open(
    os.environ.get("CONFIG_YAML")
    or f"{os.path.dirname(__file__)}/../config/config.yaml",
    "r",
) as file:
    main_config = yaml.safe_load(file)

def calculate_metrics(df: DataFrame = None, config: dict = None):
    config = config or main_config.get(calculate_metrics.__name__)
    df = (
        df
        if df is not None
        else (
            read_dataframe(config.get("input_path"))
            if config.get("input_path")
            else None
        )
    )
    df['rouge_scores'] = df.apply(calculate_rouge_scores, axis=1)
    df['meteor_scores'] = df.apply(calculate_meteor_scores, axis=1)
    df['bleu_scores'] = df.apply(calculate_bleu_scores, axis=1)
    
    output_path = config.get("output_path")
    if output_path:
        write_dataframe(df, config.get("output_path"))
    return df

        
# Example DataFrame
# data = {
#     'prediction': ['the quick brown fox', 'jumps over the lazy dog'],
#     'answer': ['quick brown fox', 'jumps over lazy dog']
# }
# df = pd.DataFrame(data)

# Initialize the ROUGE scorer
rouge = Rouge()
nltk.download('wordnet')
nltk.download('omw-1.4')


# Function to calculate BLEU scores
def calculate_bleu_scores(row):
    # Tokenizing the sentences
    reference = [row['prediction'].split()]
    candidate = row['answer'].split()
    
    # Computing BLEU score
    score = sentence_bleu(reference, candidate)
    return score

# Function to calculate ROUGE scores
def calculate_rouge_scores(row):
    scores = rouge.get_scores(row['answer'], row['prediction'])
    return scores[0]  # get_scores returns a list of scores for each pair

# Function to calculate METEOR scores
def calculate_meteor_scores(row):
    score = meteor_score([row['prediction']], row['answer'])
    return score



