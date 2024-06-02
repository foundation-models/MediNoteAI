import numpy as np
from sentence_transformers import SentenceTransformer
import onnxruntime

# Load the ONNX model
model_path = "/mnt/models/gte-large-onnx"
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
sentences = ["This is a test sentence", "Another sentence for testing"]
embeddings = encode(sentences)
print(embeddings)
print(embeddings.shape)