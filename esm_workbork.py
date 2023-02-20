# Databricks notebook source
! tensorflow --version

# COMMAND ----------

! pip install torch torchvision tensorflow accelerate transformers datasets huggingface_hub evaluate datasets requests pandas sklearn

# COMMAND ----------



# COMMAND ----------

#model_checkpoint = "facebook/esm2_t12_35M_UR50D"

# COMMAND ----------

import requests
url2="https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Cprotein_name%2Ccc_function%2Corganism_name%2Csequence%2Clength&format=tsv&query=%28%28keyword%3AKW-0800%29%29%20AND%20%28length%3A%5B1%20TO%20200%5D%29%20AND%20%28existence%3A1%29"

# COMMAND ----------

uniprot_request2 = requests.get(url2)

# COMMAND ----------

from io import BytesIO
import pandas

# COMMAND ----------

bio2 = BytesIO(uniprot_request2.content)

df2 = pandas.read_csv(bio2, compression='gzip', sep='\t')
df2

# COMMAND ----------


import gc, torch
from accelerate import Accelerator
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

from transformers import AutoTokenizer, EsmForProteinFolding, EsmModel

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
accelerator = Accelerator()
device = accelerator.device
model = model.to(device)

# COMMAND ----------


model.esm = model.esm.half()
torch.backends.cuda.matmul.allow_tf32 = True
model.trunk.set_chunk_size(64)

# COMMAND ----------

testseq=df2["Sequence"][0]

# COMMAND ----------

test_input=tokenizer([df2["Sequence"][0]], return_tensors="pt", add_special_tokens=False)['input_ids']
test_input = test_input.cuda()

# COMMAND ----------

import torch

with torch.no_grad():
    output = model(test_input)

# COMMAND ----------

output

# COMMAND ----------


