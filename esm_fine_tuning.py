# Databricks notebook source
!python --version

# COMMAND ----------

! pip install torch seaborn torchvision tensorflow biopython accelerate transformers datasets huggingface_hub evaluate requests pandas sklearn

# COMMAND ----------

model_checkpoint = "facebook/esm2_t12_35M_UR50D"

# COMMAND ----------

import requests
from io import BytesIO
import pandas as pd
import gc, torch
from accelerate import Accelerator
from Bio import SeqIO 
from Bio.SeqRecord import SeqRecord
import seaborn as sns
gc.collect()
torch.cuda.empty_cache()
accelerator = Accelerator()
device = accelerator.device

# COMMAND ----------

# MAGIC %md
# MAGIC For FASTA files from genbank's IPG

# COMMAND ----------

#ipg_nottox = []
#for seq_record in SeqIO.parse("ipg_refseq_not_toxins_not_virus_length4to500.fasta", "fasta"):
#  s=str(seq_record.seq)
#  o=seq_record.description.split('[', 1)[1].split(']')[0]
#  i=seq_record.id
#  d=seq_record.description.split(i, 1)[1].split(o)[0]
#  ipg_nottox.append([i,o,s,d])
#ipg_nottox=pd.DataFrame(ipg_nottox,columns=['id','organism','sequence','description'])
#ipg_tox = []
#for seq_record in SeqIO.parse("ipg_refseq_toxins_not_virus_length4to500.fasta", "fasta"):
#  s=str(seq_record.seq)
#  o=seq_record.description.split('[', 1)[1].split(']')[0]
#  i=seq_record.id
#  d=seq_record.description.split(i, 1)[1].split(o)[0]
#  ipg_tox.append([i,o,s,d])
#ipg_tox=pd.DataFrame(ipg_nottox,columns=['id','organism','sequence','description'])

# COMMAND ----------

#ipgtoxin_seq=ipg_tox['sequence'].tolist()
#ipgtoxin_labels=[1 for seq in ipgtoxin_seq]
#ipgnottoxin_seq=ipg_nottox['sequence'].tolist()
#ipgnottox_labels=[0 for seq in ipgnottoxin_seq]

# COMMAND ----------

# MAGIC %md
# MAGIC TSV files from uniprot/uniref

# COMMAND ----------

currentbatchtotal=134500
df0=pd.read_csv('/content/ntref50-4to500#0.tsv',sep='\t')
ntref50=pd.DataFrame(index=range(int(currentbatchtotal-(currentbatchtotal/500)+1)),columns=df0.columns)
ntref50.iloc[:500]=df0
lenlist=[]
for i in range (1,int(currentbatchtotal/500)):
  df=pd.read_csv('/content/ntref50-4to500#'+str((i*500))+'.tsv',sep='\t')
  ntref50.iloc[(i*500)-(i-1):((i+1)*500)-i]=df
tref50=pd.read_csv('/content/tref50.tsv',sep='\t')

# COMMAND ----------

len(ntref50)

# COMMAND ----------

len(ntref50.dropna())

# COMMAND ----------

# MAGIC %md
# MAGIC Export combined data from batch files so it can be imported later without needing to iterate through files again

# COMMAND ----------

data_dir = '/content/drive/My Drive/Colab Notebooks/datasets/'

# COMMAND ----------

ntref50.to_csv(data_dir+'ntref50_134500.csv')
tref50.to_csv(data_dir+'tref50.csv')

# COMMAND ----------

ntref50=pd.read_csv(data_dir+'ntref50_134500.csv')
tref50=pd.read_csv(data_dir+'tref50.csv')

# COMMAND ----------

ntref50_sequences=ntref50['Reference sequence'].tolist()
ntref50_labels=[1 for seq in ntref50_sequences]
tref50_sequences=tref50['Reference sequence'].tolist()
tref50_labels=[0 for seq in tref50_sequences]

# COMMAND ----------

sns.set_theme()
sns.barplot([len(ntref50_labels),len(tref50_labels)])

# COMMAND ----------

for seq_record in SeqIO.parse(data_dir+"bontx.fasta", "fasta"):
  s=str(seq_record.seq)

# COMMAND ----------

print(s)

# COMMAND ----------

s in set(ntref50_sequences)

# COMMAND ----------

s in set(tref50_sequences)

# COMMAND ----------

len(ntref50)==int(currentbatchtotal-(currentbatchtotal/500)+1)

# COMMAND ----------

sequences=ntref50_sequences+tref50_sequences
labels = tref50_labels+ntref50_labels
len(labels)==len(sequences)

# COMMAND ----------

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
#model = model.to(device)


# COMMAND ----------

from sklearn.model_selection import train_test_split

train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25, shuffle=True)

# COMMAND ----------

traindf=pd.DataFrame([train_sequences,train_labels],["sequences","labels"])
testdf=pd.DataFrame([test_sequences,test_labels],["sequences","labels"])
traindf.to_csv(data_dir+'ref50_134500_train_seq.csv')
testdf.to_csv(data_dir+'ref50_134500_test_seq.csv')

# COMMAND ----------

print(tokenizer.model_max_length)
max([len(seq) for seq in sequences])

# COMMAND ----------

from datasets import Dataset
train_tokens=tokenizer(train_sequences)
test_tokens=tokenizer(test_sequences)
train_dataset = Dataset.from_dict(train_tokens)
test_dataset = Dataset.from_dict(test_tokens)

# COMMAND ----------

train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)
# I dont think I need to do this unless I write my own training loop, I think HF uses gpu automatically
#train_dataset=train_dataset.with_format('torch')
#test_dataset=test_dataset.with_format('torch')
#train_dataset.to(device)
#test_dataset.to(device)

# COMMAND ----------

# MAGIC %md
# MAGIC A custom callback I was thinking of modifying to help with dynamic plots as the model was trained but am still deciding whether to use it or not.

# COMMAND ----------

from transformers import DefaultFlowCallback

# COMMAND ----------

class AccuracyCallback(DefaultFlowCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

# COMMAND ----------

# MAGIC %md
# MAGIC Arguments for the model, using standard hyperparameters here may change depending on performance. Only using 3 epochs since I think this is what I can get away with before colab kicks me off the GPU.

# COMMAND ----------

!pip install wandb -qqq
import wandb
wandb.login()

# COMMAND ----------

# MAGIC %env WANDB_PROJECT=esm-fine-tuning-toxins

# COMMAND ----------

from evaluate import load
import numpy as np

metric = load("accuracy")
trainingaccuracy=[]
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy=metric.compute(predictions=predictions, references=labels)
    trainingaccuracy.append(accuracy)
    return accuracy

# COMMAND ----------

model_name = model_checkpoint.split("/")[-1]
batch_size = 8

args = TrainingArguments(
    f"{model_name}-finetuned-2classtoxin",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to=['wandb'],
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# COMMAND ----------

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# COMMAND ----------

#trainer.add_callback(CustomCallback(trainer)) 
train = trainer.train()

# COMMAND ----------

trainer.evaluate()

# COMMAND ----------

model_dir = '/content/drive/My Drive/Colab Notebooks/models/'

# COMMAND ----------

trainer.save_model(model_dir + 'esm_fine_tuning3epochstrainer')
model.save_pretrained(model_dir + 'esm_fine_tuning3epochsmodel')

# COMMAND ----------



# COMMAND ----------

model = AutoModelForSequenceClassification.from_pretrained(model_dir + 'esm_fine_tuning3epochsmodel', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# COMMAND ----------

bontx_encoded=tokenizer(s,return_tensors='pt')
bontx_output=model(**bontx_encoded)

# COMMAND ----------

from torch import nn
probabilities = nn.functional.softmax(bontx_output.logits, dim=-1)
print(probabilities)

# COMMAND ----------


