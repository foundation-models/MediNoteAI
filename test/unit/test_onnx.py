import numpy as np
from sentence_transformers import SentenceTransformer
import onnxruntime

# Load the ONNX model
model_path = "/mnt/models/gte-large-onnx/model.onnx"
onnx_path = "/mnt/models/gte-large-onnx/model.onnx "

from transformers import Pipeline
import torch.nn.functional as F
import torch 

# copied from the model card
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SentenceEmbeddingPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        # we don't have any hyperameters to sanitize
        preprocess_kwargs = {}
        return preprocess_kwargs, {}, {}
      
    def preprocess(self, inputs):
        encoded_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        return encoded_inputs

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return {"outputs": outputs, "attention_mask": model_inputs["attention_mask"]}

    def postprocess(self, model_outputs):
        # Perform pooling
        sentence_embeddings = mean_pooling(model_outputs["outputs"], model_outputs['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
onnx_session = onnxruntime.InferenceSession(model_path)

# Define a helper function to run inference
def encode(sentences):
    # Tokenize the input sentences
    inputs = tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True)
    
    # Run inference with ONNX Runtime
    ort_inputs = {onnx_session.get_inputs()[0].name: inputs['input_ids'],
                  onnx_session.get_inputs()[1].name: inputs['attention_mask']}
    output = onnx_session.run(None, ort_inputs)[0]
    
    return output

# Example usage
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model = ORTModelForFeatureExtraction.from_pretrained(onnx_path, file_name="model_optimized_quantized.onnx")
tokenizer = AutoTokenizer.from_pretrained(onnx_path)

q8_emb = SentenceEmbeddingPipeline(model=model, tokenizer=tokenizer)

pred = q8_emb("Could you assist me in finding my lost card?")
print(pred[0][:5])