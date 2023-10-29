from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModel
import os
import torch
import math
from tqdm.auto import tqdm
import gc
@torch.no_grad()
def embed_datasets(input_path="data/clean", pretrained_model_path='BAAI/bge-large-en-v1.5', num_tokens=512, num_text_chunks_per_dataset=100, output_dir="data/dataset"):
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict = {"text": [], "embedding": [], "tokens": [], "database": [], "page": [], "chunk": []}
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    model = AutoModel.from_pretrained(pretrained_model_path).to("cuda")
    model.eval()
    for subtask in os.listdir(input_path):
        subtask_path = os.path.join(input_path, subtask)
        output_subtask_path = os.path.join(output_dir, subtask)
        os.makedirs(output_subtask_path, exist_ok=True)
        prev_save_idx = -1
        for idx, file in tqdm(enumerate(os.listdir(subtask_path))):
            torch.cuda.empty_cache()
            gc.collect()
            file_path = os.path.join(subtask_path, file)
            with open(file_path, "r", encoding='utf-8') as f:
                text = f.read()
            encoded_input = tokenizer([text], padding=True, truncation=False, return_tensors='pt', add_special_tokens=True)

            num_tokens_in_text = encoded_input['input_ids'].shape[1]
            num_token_chunks = math.ceil(num_tokens_in_text / num_tokens)
            token_chunks = []
            for i in range(num_token_chunks):
                token_chunk =encoded_input["input_ids"][:, num_tokens*i:num_tokens*(i+1)]
                token_input_ids = token_chunk
                token_chunks.append(token_input_ids)

            embeddings = []
            for i in range(num_token_chunks):
                input_ids = token_chunks[i]
                token_type_ids = torch.zeros_like(input_ids)
                attention_masks = torch.ones_like(input_ids)
                encoded_chunk = {
                    "input_ids": input_ids.to("cuda"),
                    "token_type_ids": token_type_ids.to("cuda"),
                    "attention_mask": attention_masks.to("cuda")
                }
                model_output = model(**encoded_chunk)
                sentence_embeddings = model_output[0][:, 0]
                embeddings.append(sentence_embeddings.cpu().numpy()[0])
            i = 0
            for (token_chunk, embedding) in zip(token_chunks, embeddings):
                dataset_dict["text"].append(tokenizer.decode(token_chunk[0]))
                dataset_dict["tokens"].append(token_chunk[0].cpu().numpy())
                dataset_dict["embedding"].append(embedding)
                dataset_dict["database"].append(subtask)
                dataset_dict["file"].append(file)
                dataset_dict["chunk"].append(i)
                i += 1

            if (idx + 1) % num_text_chunks_per_dataset == 0:
                dataset_idx = idx // num_text_chunks_per_dataset
                prev_save_idx = dataset_idx
                dataset = Dataset.from_dict(dataset_dict)
                dataset.save_to_disk(os.path.join(output_subtask_path, str(dataset_idx)))
                for key in dataset_dict:
                    dataset_dict[key] = []
        dataset_idx = prev_save_idx + 1
        dataset = Dataset.from_dict(dataset_dict)
        dataset.save_to_disk(os.path.join(output_subtask_path, str(dataset_idx)))
        for key in dataset_dict:
            dataset_dict[key] = []
def combine_datasets(output_dir="data/dataset", output_dataset_path="data/scraped_dataset"):
    datasets = []
    for subtask in os.listdir(output_dir):
        subtask_path = os.path.join(output_dir, subtask)
        for dataset_name in os.listdir(subtask_path):
            dataset_path = os.path.join(subtask_path, dataset_name)
            dataset = load_from_disk(dataset_path)
            datasets.append(dataset)
    dataset = concatenate_datasets(datasets)
    dataset.save_to_disk(output_dataset_path)
if __name__ == "__main__":
    output_dir = "data/full_token_dataset"
    output_dataset_path = "data/full_token_scaped_dataset"
    hf_dataset_path = "Isamu136/penetration_testing_scraped_dataset"

    embed_datasets(output_dir=output_dir)
    combine_datasets(output_dir=output_dir, output_dataset_path=output_dataset_path)
    dataset = load_from_disk(output_dataset_path)
    dataset.push_to_hub(hf_dataset_path)