from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
model = AutoModel.from_pretrained("thenlper/gte-small")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

text = "This is a test."
encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
with torch.no_grad():
    model_output = model(**encoded_input)

embedding = mean_pooling(model_output, encoded_input['attention_mask'])
print("Embedding shape:", embedding.shape)
