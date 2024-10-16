from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

sentences = ['That is a happy person', 'That is a very happy person']

model = SentenceTransformer('/mnt/models/gte-large-en-v1.5', trust_remote_code=True)
embeddings = model.encode(sentences)
print(cos_sim(embeddings[0], embeddings[1]))
