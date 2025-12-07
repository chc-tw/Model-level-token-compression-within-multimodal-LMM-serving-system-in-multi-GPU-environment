import evaluate
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
file_path = "/storage/ice1/9/1/cho322/research3/experiments/sharegpt4o_image_caption/dynamic_compression-trace-3-rep-1/Trace with Peak QPS_10.json"

with open(file_path, "r") as f:
    data = json.load(f)

predictions = [m["generated_text"] for m in data["individual_request_metrics"]]
references = [m["answer_text"] for m in data["individual_request_metrics"]]

# 2. Calculate embeddings by calling model.encode()
emb_p = model.encode(predictions)
emb_r = model.encode(references)

similarities = model.similarity(emb_p, emb_r)
# compute average similarity on the diagonal
avg_similarity = similarities.diagonal().mean()
print(avg_similarity)
