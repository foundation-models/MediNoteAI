import nltk
from MediNoteAI.src.medinote.metrics.metrics import Rouge
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu

# Example DataFrame
# data = {
#     'correct_answer': ['the quick brown fox', 'jumps over the lazy dog'],
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
    reference = [row['correct_answer'].split()]
    candidate = row['answer'].split()
    
    # Computing BLEU score
    score = sentence_bleu(reference, candidate)
    return score

# Function to calculate ROUGE scores
def calculate_rouge_scores(row):
    scores = rouge.get_scores(row['answer'], row['correct_answer'])
    return scores[0]  # get_scores returns a list of scores for each pair

# Function to calculate METEOR scores
def calculate_meteor_scores(row):
    score = meteor_score([row['correct_answer']], row['answer'])
    return score

def calculate_metrics(df):
    # Apply the function to each row
    df['rouge_scores'] = df.apply(calculate_rouge_scores, axis=1)
    df['meteor_scores'] = df.apply(calculate_meteor_scores, axis=1)
    df['bleu_scores'] = df.apply(calculate_bleu_scores, axis=1)
    return df


