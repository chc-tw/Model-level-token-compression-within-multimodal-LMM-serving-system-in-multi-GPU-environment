import evaluate
import json
import argparse
from sentence_transformers import SentenceTransformer
import optparse
import os

def main(file_path: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")

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

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--file_path", type=str, required=True)
    args = args.parse_args()
    main(args.file_path)