


import os
from pandas import DataFrame
from medinote.curation.util import generate_training_dataset


def test_generate_training_dataset():
    # Example DataFrame (replace this with your actual DataFrame)
    data = {
        'brief_draft': ['Draft 1', 'Draft 2', 'Draft 3'],
        'narrative': ['Narrative 1', 'Narrative 2', 'Narrative 3']
    }
    df = DataFrame(data)
    os.environ['INPUT_COLUMN'] = 'brief_draft'
    os.environ['OUTPUT_COLUMN'] = 'narrative'
    
    
    generate_training_dataset(df)