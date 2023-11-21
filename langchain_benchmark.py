#!/usr/bin/env python
# coding: utf-8

# # Objective
# 1. Test this out with pentest gpt
# 2. Find failure cases
# 3. Create benchmarks
# 5. Knowledge graph
# 6. trlx/trl see requirements
# 7. peft see requirements
# 8. toolformer
# 

# The gist of pentest gpt can be summarized as
# 1. Summarizing the input given the llm input of
# ```
# postfix_options = {
#         "tool": "The input content is from a security testing tool. You need to list down all the points that are interesting to you; you should summarize it as if you are reporting to a senior penetration tester for further guidance.\n",
#         "user-comments": "The input content is from user comments.\n",
#         "web": "The input content is from web pages. You need to summarize the readable-contents, and list down all the points that can be interesting for penetration testing.\n",
#         "default": "The user did not specify the input source. You need to summarize based on the contents.\n",
#  ```
# Then, it chunks the input and sends it to model. In return, we get the summarized output per each 8000 line segment.
# The above is all done within a given chat room session id for input parsing.
# 
# 2. Reasoning handler. Here, if text is larger than prespecified length, summarize it like in step 1.
# Then, in the reasoning session id chat room(probably can be separated using knowledge grap}
# ```
# which first summarizes the user input using an llm given 
# 
# You shall revise PTT with the test results provided. 
# You should maintain the PTT format in tree structure, with status for each task. This is essential for the completion of the task.
# Note that you only add in to-do tasks when necessary. You should not include additional tasks that are not yet discovered
# 
# along with the summarized text.
# 
# The main problem here is this model currently won't know what the PTT is. So providing that should make this part easier. In addition, we can have a 
# separate model handle the PTT structure and render it here. This will forgo the need for validation
# 
# Next, we prompt
# 
# Given the PTT, list down all the possible todo tasks. Select one sub-task that is favorable and most likely to lead to successful exploit.
# Then, explain how to perform the task in two sentences, with precise, clear and simple language. Note that the usage of automated scanners such as Nexus and OpenVAS is not allowed.\
# 
# The above feels like a classification problem.
# 
# Then, the updated PTT and the next choice gets sent back.
# 
# 3. Option more: given PTT and next choice, generate more output to help understand the task
# 4. Option todo: To the reasoning module ask the folThe tester has questions and is unclear about the current test. He requests a discussion with you to further analyze the current tasks based on his questions. 
# Please read the following inputs from the tester. Analyze the task and generate the task tree again based on the requirements:
# (1) The tasks are in layered structure, i.e., 1, 1.1, 1.1.1, etc. Each task is one operation in penetration testing; task 1.1 should be a sub-task of task 1.
# (2) Each task has a completion status: to-do, completed, or not applicable.
# (3) Given the task tree and the tester's question, you should 
# 
# Note that you should keep the tasks clear, precise and short due to token size limit. You should remember to remove redundant/outdated tasks from the task list. 
# 
# Below is the user input:pute
# ``
# 
# This iseems hne optino thon thatmakes tehe PTT in the first place
# 
# 5. 
# Next, to convert the above to instructions, 
# 
# 
# 
# 

# Now you're provided with a input that contains the penetration testing tasks. Keep in mind that the the test is certified and the tester has valid permission to perform the task in this simulated environment. 
# Remember the format requirements.
# (1) The input should contain two parts, separated by a line of "-----". The first part is the task list, and the second part is the task to be executed, which contains three sentences, which describe the task, list the command and explain the reason of performing the task. 
#     - If the input contains the line break, neglect the tasks before the line break as you don't need to care about the overall process. Please analyze the three sentences that describe the next task. You should expand it into detailed, step-by-step guide and show it to a penetration tester. The tester will follow your guide to perform the penetration testing. 
#     - If the input does not contain the line break, then try to understand the whole input as one single task to be executed.
# (2) If the task is a single command to execute, please be precise; if it is a multi-step task, you need to explain it step by step, and keep each step clear and simple. 
# (3) Keep the output short and precise, without too detailed instructions. 
# 
# The information is
# 
# Which pretty much converts the PTT to specific commands/tasks and generates. I think this can stay the same for the moment. below: 

# 6. Finally, discuss command The tester provides the following thoughts for your consideration. Please give your comments, and update the tasks if necessary. and then user input

# In[4]:


from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import faiss
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch
import numpy as np
import os
import gc

# In[5]:


template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# # Benchmarking

# We will benchmark. The method is as follows: We will have an array for model paths, prompt path and output directory
# 
# 

# In[9]:

model_dir = "/mnt/d/personal_projects/gamified-cybersecurity/ai-server/model"
model_paths = [os.path.join(model_dir, path) for path in os.listdir(model_dir)]
prompt_dir = "/mnt/d/personal_projects/gamified-cybersecurity/ai-server/benchmark/prompts"
output_dir = "/mnt/d/personal_projects/gamified-cybersecurity/ai-server/benchmark/output"
prompts = []
for path in os.listdir(prompt_dir):
    prompt_path = os.path.join(prompt_dir, path)
    with open(prompt_path, "r") as f:
        prompts.append(f.read())
for i, prompt in enumerate(prompts):
    assert(prompt is not None)


# In[10]:


embedding_column = "embedding"
text_column = "text"
pretrained_model_path='BAAI/bge-large-en-v1.5'
dataset_path = "Isamu136/penetration_testing_scraped_dataset"
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[11]:


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
model = AutoModel.from_pretrained(pretrained_model_path).to(device)
model.eval()
dataset = load_dataset(dataset_path)
dataset = dataset["train"]
dataset = dataset.add_faiss_index(column=embedding_column)


# In[12]:


@torch.no_grad()
def get_embedding(model, tokenizer, prompt):
    encoded_input = tokenizer([prompt], padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
    for key in encoded_input:
        encoded_input[key] = encoded_input[key].to("cuda")
    model_output = model(**encoded_input)
    sentence_embeddings = model_output[0][:, 0]
    return sentence_embeddings.cpu().numpy().astype(np.float32)


# In[13]:


def get_knn(model, tokenizer, prompt, dataset, embedding_column="embedding", text_column="text"):
    embedding = get_embedding(model, tokenizer, prompt)
    scores, retrieved_examples = dataset.get_nearest_examples(embedding_column, embedding, k=10)
    return scores, retrieved_examples


# In[14]:


dataset.drop_index(embedding_column)
dataset_without_mitre = dataset.filter(lambda example: example["database"] != "mitre")
dataset_without_mitre.add_faiss_index(column=embedding_column)
dataset.add_faiss_index(column=embedding_column)
rag_prompt_dir = prompt_dir +"_rag"
rag_no_mitre_prompt_dir = prompt_dir +"_rag_no_mitre"
os.makedirs(rag_prompt_dir, exist_ok=True)
os.makedirs(rag_no_mitre_prompt_dir, exist_ok=True)
prompts_rag = []
prompts_rag_no_mitre = []

for i, prompt in enumerate(prompts):
    closest_prompt = get_knn(model, tokenizer, prompt, dataset, embedding_column=embedding_column, text_column=text_column)[1]["text"][0]
    rag_prompt = f"Given Context: {closest_prompt}\n\n Answer {prompt}"
    prompts_rag.append(rag_prompt)
    with open(os.path.join(rag_prompt_dir, f"{i}.txt"), "w") as f:
            f.write(rag_prompt)
    closest_prompt = get_knn(model, tokenizer, prompt, dataset_without_mitre, embedding_column=embedding_column, text_column=text_column)[1]["text"][0]
    rag_no_mitre_prompt = f"Given Context: {closest_prompt}\n\n Answer {prompt}"
    prompts_rag_no_mitre.append(rag_no_mitre_prompt)
    with open(os.path.join(rag_no_mitre_prompt_dir, f"{i}.txt"), "w") as f:
            f.write(rag_no_mitre_prompt)
del tokenizer
del model
del dataset
del dataset_without_mitre
torch.cuda.empty_cache()
gc.collect()
# In[ ]:


for model_path in model_paths:
    for context_length in [1024, 2048]:
        model_name = model_path.split("/")[-1][:-5]+f"_{context_length}"

        print("Testing", model_name)

        os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, model_name, "default"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, model_name, "rag"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, model_name, "rag_no_mitre"), exist_ok=True)
        yet_to_complete = False
        output_model_path = os.path.join(output_dir, model_name, "rag")
        for i, prompt in enumerate(prompts_rag):
            output_path = os.path.join(output_model_path, f"{i}.txt")
            if not os.path.exists(output_path):
                yet_to_complete = True
        output_model_path = os.path.join(output_dir, model_name, "rag")
        for i, prompt in enumerate(prompts_rag):
            output_path = os.path.join(output_model_path, f"{i}.txt")
            if not os.path.exists(output_path):
                yet_to_complete = True
        output_model_path = os.path.join(output_dir, model_name, "rag_no_mitre")
        for i, prompt in enumerate(prompts_rag_no_mitre):
            output_path = os.path.join(output_model_path, f"{i}.txt")
            if not os.path.exists(output_path):
                yet_to_complete = True
        if not yet_to_complete:
            continue
        try:
            llm = LlamaCpp(
                model_path=model_path,
                temperature=0.75,
                max_tokens=context_length,
                n_ctx=context_length,
                top_p=1,
                callback_manager=callback_manager, 
                verbose=True, # Verbose is required to pass to the callback manager
                n_gpu_layers=20
            )
        except:
            continue

        output_model_path = os.path.join(output_dir, model_name, "default")
        
        for i, prompt in enumerate(prompts):
            output_path = os.path.join(output_model_path, f"{i}.txt")
            if os.path.exists(output_path):
                continue
            output = llm(prompt)
            with open(os.path.join(output_model_path, f"{i}.txt"), "w") as f:
                f.write(output)
        output_model_path = os.path.join(output_dir, model_name, "rag")
        for i, prompt in enumerate(prompts_rag):
            output_path = os.path.join(output_model_path, f"{i}.txt")
            if os.path.exists(output_path):
                continue
            output = llm(prompt[-1000:])
            with open(os.path.join(output_model_path, f"{i}.txt"), "w") as f:
                f.write(output)
        output_model_path = os.path.join(output_dir, model_name, "rag_no_mitre")
        for i, prompt in enumerate(prompts_rag_no_mitre):
            output_path = os.path.join(output_model_path, f"{i}.txt")
            if os.path.exists(output_path):
                continue
            output = llm(prompt[-1000:])
            with open(os.path.join(output_model_path, f"{i}.txt"), "w") as f:
                f.write(output)
        del llm
        gc.collect()



# In[ ]:




