# Databricks notebook source
!python --version

# COMMAND ----------

! pip install torch seaborn torchvision tensorflow biopython accelerate transformers datasets huggingface_hub evaluate requests pandas sklearn optuna ray[tune] hyperopt nvidia-ml-py3 wandb -qqq

# COMMAND ----------

model_checkpoint = "facebook/esm2_t6_8M_UR50D"
model_name = model_checkpoint.split("/")[-1]
#model_checkpoint = "facebook/esm2_t12_35M_UR50D"
#model_checkpoint = "facebook/esm2_t30_150M_UR50D"
#model_checkpoint = "facebook/esm2_t33_650M_UR50D"

# COMMAND ----------

import requests
import math
from io import BytesIO
import pandas as pd
import gc, torch
from accelerate import Accelerator
from Bio import SeqIO 
from Bio.SeqRecord import SeqRecord
import seaborn as sns
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, logging
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.utils.random import sample_without_replacement
gc.collect()
torch.cuda.empty_cache()
accelerator = Accelerator()
device = accelerator.device

# COMMAND ----------

from google.colab import drive
drive.mount('/content/drive')

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
data_dir = '/content/drive/My Drive/Colab Notebooks/datasets/'
model_dir = '/content/drive/My Drive/Colab Notebooks/models/'
run_number=1
ratio=1

# COMMAND ----------

for seq_record in SeqIO.parse(data_dir+"bontx.fasta", "fasta"):
  s=str(seq_record.seq)

# COMMAND ----------

def split(tox_seq,nottox_seq):
  sequences=tox_seq+nottox_seq
  labels = [1 for i in tox_seq]+[0 for i in nottox_seq]
  train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25, shuffle=True)
  return train_sequences, test_sequences, train_labels, test_labels

def tokenize_seqs(train_sequences, test_sequences):
  train_tokens=tokenizer(train_sequences)
  test_tokens=tokenizer(test_sequences)
  return train_tokens, test_tokens

def save_seq_splits(train_sequences, test_sequences, train_labels, test_labels,data_dir,ratio,trial_number,run_number):
  traindf=pd.DataFrame({"sequences":train_sequences,"labels":train_labels})
  testdf=pd.DataFrame({"sequences":test_sequences,"labels":test_labels})
  traindf.to_csv(data_dir+f"train-r:{ratio}tn:{trial_number}rn:{run_number}.csv")
  testdf.to_csv(data_dir+f"test-r:{ratio}tn:{trial_number}rn:{run_number}.csv")

def from_token_splits(data_dir,ratio,trial_number,run_number):
  train_tokens = pd.read_csv(data_dir+f"train-r:{ratio}tn:{trial_number}rn:{run_number}.csv")["tokens"].tolist()
  test_tokens = pd.read_csv(data_dir+f"test-r:{ratio}tn:{trial_number}rn:{run_number}.csv")["tokens"].tolist()
  train_labels = pd.read_csv(data_dir+f"train-r:{ratio}tn:{trial_number}rn:{run_number}.csv")["labels"].tolist()
  test_labels = pd.read_csv(data_dir+f"test-r:{ratio}tn:{trial_number}rn:{run_number}.csv")["labels"].tolist()
  return train_tokens, test_tokens, train_labels, test_labels

def create_datasets(train_tokens, test_tokens, train_labels, test_labels):
  train_dataset = Dataset.from_dict(train_tokens)
  test_dataset = Dataset.from_dict(test_tokens)
  train_dataset = train_dataset.add_column("labels", train_labels)
  test_dataset = test_dataset.add_column("labels", test_labels)
  return train_dataset, test_dataset



def train_test_pipe(data_dir,ratio=1,trial_number=1,run_number=1,uniref_level=50,load_saved=False,output_class_lengths=False):
  if (load_saved==False):
    tox_seq=pd.read_csv(data_dir+f'tref{uniref_level}.csv')['Reference sequence'].tolist()
    nottox_seq=pd.read_csv(data_dir+f'ntref{uniref_level}-1trial-{ratio}:1-seq.csv')[f'trial{trial_number}'].tolist()
    train_sequences, test_sequences, train_labels, test_labels = split(tox_seq,nottox_seq)
    train_tokens, test_tokens = tokenize_seqs(train_sequences, test_sequences)
    save_seq_splits(train_sequences, test_sequences, train_labels, test_labels,data_dir,ratio,trial_number,run_number)
    train_dataset, test_dataset=create_datasets(train_tokens, test_tokens, train_labels, test_labels)
  elif (load_save==True):
    train_sequences, test_sequences, train_labels, test_labels = from_token_splits(data_dir,ratio,trial_number,run_number)
    train_tokens, test_tokens = tokenize_seqs(train_sequences, test_sequences)
    train_dataset, test_dataset=create_datasets(train_tokens, test_tokens, train_labels, test_labels)
  if (output_class_lengths==False):
    tox_seq=pd.read_csv(data_dir+f'tref{uniref_level}.csv')['Reference sequence'].tolist()
    nottox_seq=pd.read_csv(data_dir+f'ntref{uniref_level}-{ratio}:1-seq.csv')[f'trial{trial_number}'].tolist()
    seq_counts=pd.DataFrame([['not toxin',len(ntref50_labels)],['toxin',len(tref50_labels)]],columns=["sequence type","data count"])
    print(seq_counts)
  return train_dataset, test_dataset

# COMMAND ----------

train, test=train_test_pipe(data_dir=data_dir,ratio=ratio,trial_number=1,run_number=run_number,uniref_level=50,output_class_lengths=True)

# COMMAND ----------

# MAGIC %env WANDB_PROJECT=esm-fine-tuning-toxins
# MAGIC %env WANDB_NOTEBOOK_NAME=esm_fine_tuning_hyperparamsearch_2class
# MAGIC %env WANDB_WATCH=all

# COMMAND ----------

import wandb
wandb.login()

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

from transformers import TrainerCallback
class hyperparamcallback(TrainerCallback):
  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.args=args
  def on_train_begin(self, args, state, control, **kwargs):
        run=f"{model_name}-{ratio}:1-run:{run_number}-{args.num_train_epochs}-{args.learning_rate}-{args.per_device_train_batch_size}"
        print("Starting training:"+run)
  def on_train_end(self, args, state, control, **kwargs):
        print("Ending training:"+run)

# COMMAND ----------

def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=2,return_dict=True)

# COMMAND ----------

training_args = TrainingArguments(
    overwrite_output_dir =True,
    evaluation_strategy = 'steps',
    eval_steps=500,
    warmup_ratio=0.01,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=5,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    fp16=True,
    weight_decay=0.01,
    run_name=f"{model_name}",
    output_dir=f"/content/drive/My Drive/Colab Notebooks/models/hyperopt2class/{model_name}",
    report_to=['wandb'],
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# COMMAND ----------

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.run_name=f"{model_name}-{ratio}:1-run:{run_number}-{self.args.num_train_epochs}-{self.args.learning_rate}-{self.args.per_device_train_batch_size}"
        self.args.output_dir=f"/content/drive/My Drive/Colab Notebooks/models/hyperopt2class/{model_name}-{ratio}:1-run:{run_number}-{self.args.num_train_epochs}-{self.args.learning_rate}-{self.args.per_device_train_batch_size}"

# COMMAND ----------

from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
#EarlyStoppingCallback(early_stopping_patience=3,early_stopping_threshold=0.02)
trainer = CustomTrainer(
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    model_init=model_init,
    #data_collator=DataCollatorWithPadding,
    callbacks = [hyperparamcallback]

)

# COMMAND ----------

def searchspace(trial):
    from ray import tune

    return {
        "learning_rate": tune.loguniform(2e-5, 2e-2),
        "num_train_epochs": tune.choice(range(1, 6)),
        "seed": tune.choice(range(1, 41)),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
        #"train_dataset": tune.choice(["1:1class_train","2:1class_train","5:1class_train","10:1class_train"]),
        #"eval_dataset" : "train_dataset"[0:-5]+"test"
    }

# COMMAND ----------

!pip install optuna

# COMMAND ----------

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 1e-2, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 6),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
    }

# COMMAND ----------

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=5,
)

# COMMAND ----------



# COMMAND ----------

trainer.evaluate()

# COMMAND ----------

model_dir = '/content/drive/My Drive/Colab Notebooks/models/'

# COMMAND ----------

trainer.save_model(model_dir + 'esm_fine_tuning_10epochs_trainer_earlystopping')
model.save_pretrained(model_dir + 'esm_fine_tuning_10epochs_model_earlystopping')

# COMMAND ----------

wandb.finish()

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


