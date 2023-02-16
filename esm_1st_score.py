# Databricks notebook source
! python --version

# COMMAND ----------

! pip install torch torchvision tensorflow accelerate transformers datasets huggingface_hub evaluate datasets requests pandas sklearn

# COMMAND ----------

model_checkpoint = "facebook/esm2_t12_35M_UR50D"

# COMMAND ----------

import requests
from io import BytesIO
import pandas
import gc, torch
from accelerate import Accelerator
gc.collect()
torch.cuda.empty_cache()
accelerator = Accelerator()
device = accelerator.device
url="https://rest.uniprot.org/uniref/stream?compressed=true&fields=id%2Cname%2Ctypes%2Ccount%2Corganism%2Clength%2Cidentity%2Csequence&format=tsv&query=%28%28length%3A%5B4%20TO%20500%5D%29%20AND%20toxin%29%20AND%20%28identity%3A0.5%29"
uniprot_request = requests.get(url)
bio = BytesIO(uniprot_request.content)
df = pandas.read_csv(bio, compression='gzip', sep='\t')

# COMMAND ----------

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
labels=2

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=labels)
#model.esm = model.esm.half()
#torch.backends.cuda.matmul.allow_tf32 = True
model = model.to(device)


# COMMAND ----------

test_prot=tokenizer(df["Reference sequence"][0], return_tensors="pt")
test_prot.to(device)

# COMMAND ----------

model(**test_prot)

# COMMAND ----------

import torch

with torch.no_grad():
    output = model(**test_prot)

# COMMAND ----------

from torch import nn
probabilities = nn.functional.softmax(output.logits, dim=-1)
print(probabilities)

# COMMAND ----------

prot_sequences=tokenizer(df["Reference sequence"].tolist(), return_tensors="pt",padding=True,truncation=True)
prot_sequences.to(device)

# COMMAND ----------

toxin_seq=tokenizer(df["Reference sequence"].tolist()[0:10], return_tensors="pt",padding=True,truncation=True)
toxin_seq.to(device)

# COMMAND ----------

with torch.no_grad():
    predictions = model(**toxin_seq)

# COMMAND ----------

probabilities = nn.functional.softmax(predictions.logits, dim=-1)
print(probabilities)
#seems to show correlation already, class 1 was predicted for all selected toxins.

# COMMAND ----------

url2="https://rest.uniprot.org/uniref/stream?compressed=true&fields=id%2Cname%2Ctypes%2Ccount%2Corganism%2Clength%2Cidentity%2Csequence&format=tsv&query=%28%28length%3A%5B4%20TO%20500%5D%29%20NOT%20toxin%20AND%20%28taxonomy_id%3A8570%29%29%20AND%20%28identity%3A0.5%29"
uniprot_request2 = requests.get(url2)
bio2 = BytesIO(uniprot_request2.content)
df2 = pandas.read_csv(bio2, compression='gzip', sep='\t')

# COMMAND ----------

not_toxin_seq=tokenizer(df2["Reference sequence"].tolist()[0:10], return_tensors="pt",padding=True,truncation=True)
not_toxin_seq.to(device)

# COMMAND ----------

with torch.no_grad():
    predictions2 = model(**not_toxin_seq)

# COMMAND ----------

probabilities = nn.functional.softmax(predictions2.logits, dim=-1)
print(probabilities)
# howeer, still classifies not toxins into class one as well

# COMMAND ----------


