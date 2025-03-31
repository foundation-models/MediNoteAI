import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# Each query needs to be accompanied by an corresponding instruction describing the task.
task_name_to_instruct = {"example": "Given a setence, retrieve tags that best describe matches the sentence",}

query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
queries = [
    'Comparison between YOLO and RCNN on real world videos', 
    'The beauty of the work lies in the way it architects the fundamental idea that humans look at the overall image and then individual pieces of it.',
    'A new automatic summarization task with high source compression requiring expert background knowledge and complex language understanding.'
    ]

# No instruction needed for retrieval passages
passage_prefix = ""
passages = [
    "this sentence is about natural-language-processin algorithms",
    "this is not related to any of the topics",
    "this sentence related to computer-vision algorithms",
    "this sentence realted to speech-recognition algorithms",
    
]

# load model with tokenizer
model = AutoModel.from_pretrained('/mnt/models/NV-Embed-v1', trust_remote_code=True)

# get the embeddings
max_length = 4096
query_embeddings = model.encode(queries, instruction=query_prefix, max_length=max_length)
passage_embeddings = model.encode(passages, instruction=passage_prefix, max_length=max_length)

# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

# get the embeddings with DataLoader (spliting the datasets into multiple mini-batches)
# batch_size=2
# query_embeddings = model._do_encode(queries, batch_size=batch_size, instruction=query_prefix, max_length=max_length)
# passage_embeddings = model._do_encode(passages, batch_size=batch_size, instruction=passage_prefix, max_length=max_length)

scores = (query_embeddings @ passage_embeddings.T) * 100
print(scores.tolist())
#[[77.9402084350586, 0.4248958230018616], [3.757718086242676, 79.60113525390625]]
