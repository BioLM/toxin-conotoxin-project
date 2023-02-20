# Databricks notebook source
dbutils.widgets.dropdown("DEV", "TRUE", ["TRUE", "FALSE"])
dbutils.widgets.text("MODEL_CHECKPOINT", "facebook/esm2_t12_35M_UR50D")

# Job stuff
dbutils.widgets.text("ANALYSIS_ID", "test_analysis")
dbutils.widgets.text("MAX_TRAIN_SAMPLES", "200000")
dbutils.widgets.text("MAX_VALIDATE_SAMPLES", "100000")
dbutils.widgets.text("EPOCHS", "3")


MODEL_CHECKPOINT = dbutils.widgets.get("MODEL_CHECKPOINT")
DEV = True if dbutils.widgets.get("DEV").upper() == 'TRUE' else False
ANALYSIS_ID = dbutils.widgets.get("ANALYSIS_ID").strip()
EPOCHS = int(dbutils.widgets.get("EPOCHS"))
MAX_TRAIN_SAMPLES = int(dbutils.widgets.get("MAX_TRAIN_SAMPLES"))
MAX_VALIDATE_SAMPLES = int(dbutils.widgets.get("MAX_VALIDATE_SAMPLES"))

LOCAL_TMPDIR = f'/local_disk0/{ANALYSIS_ID}'

if DEV and EPOCHS > 1:
    EPOCHS = 1
if DEV and MAX_TRAIN_SAMPLES > 5000:
    MAX_TRAIN_SAMPLES = 5000
if DEV and MAX_VALIDATE_SAMPLES > 1000:
    MAX_VALIDATE_SAMPLES = 1000

# COMMAND ----------

!python --version && mkdir -p {LOCAL_TMPDIR}

# COMMAND ----------

!pip install -U pip && pip install awscli && pip install -r ./requirements.txt

# COMMAND ----------

!mkdir -p /local_disk0/.hf_cache

# COMMAND ----------

import gc, torch
import json
import mlflow
import numpy as np
import os
import pandas as pd
import requests
import seaborn as sns
import torch

os.environ['TRANSFORMERS_CACHE'] = '/local_disk0/.hf_cache'

from ast import literal_eval
from accelerate import Accelerator
from datasets import Dataset
from evaluate import load
from io import BytesIO
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import (AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer, AutoTokenizer, pipeline,
                          TextClassificationPipeline)
from transformers import DefaultFlowCallback


n_gpus = torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in list(range(n_gpus))])

sns.set_style('whitegrid')

accelerator = Accelerator()
device = accelerator.device

device

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Development to Create Example Inputs

# COMMAND ----------

!wget -O /local_disk0/tref50_nv_4to500.tsv.gz "https://github.com/BioLM/toxin-conotoxin-project/blob/main/UniRef/UniRef_Toxins/notvirus/compressed/tref50_nv_4to500.tsv.gz?raw=true"

# COMMAND ----------

tref50 = pd.read_csv('/local_disk0/tref50_nv_4to500.tsv.gz', sep='\t')


# COMMAND ----------

print(tref50.shape)
print(tref50.dropna().shape)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Those are the `positive` class, i.e. `tref` for `ToxinsRef`.
# MAGIC 
# MAGIC And `ntref` is the `negative` class, i.e. `NotToxinRef`. Let's get those.

# COMMAND ----------

!wget -O /local_disk0/ntref50_1trial_2-1_seq.csv.gz "https://github.com/BioLM/toxin-conotoxin-project/blob/main/UniRef/UniRef_NotToxins/notvirus/50level/compressedsampled/ntref50-1trial-2:1-seq.csv.gz?raw=true"

# COMMAND ----------

ntref50 = pd.read_csv('/local_disk0/ntref50_1trial_2-1_seq.csv.gz', names=['index', 'trial'])

ntref_seqs = literal_eval(ntref50.trial.iloc[1])

ntref50 = pd.DataFrame(ntref_seqs, columns=['sequence'])

ntref50.shape

# COMMAND ----------

ntref50_sequences = ntref50['sequence'].tolist()
ntref50_labels = [1 for seq in ntref50_sequences]

tref50_sequences = tref50['Reference sequence'].tolist()
tref50_labels = [0 for seq in tref50_sequences]

print(len(tref50_labels))
print(len(ntref50_labels))

# COMMAND ----------

bontx = """
MKLEINKFNYNDPIDGINVITMRPPRHSDKINKGKGPFKAFQVIKNIWIVPERYNFTNNTNDLNIPSEPIMEADAIYNPNYLNTPSEKDEFLQGVIKVLERIKSKPEGEKLLELISSSIPLPLVSNGALTLSDNETIAYQENNNIVSNLQANLVIYGPGPDIANNATYGLYSTPISNGEGTLSEVSFSPFYLKPFDESYGNYRSLVNIVNKFVKREFAPDPASTLMHELVHVTHNLYGISNRNFYYNFDTGKIETSRQQNSLIFEELLTFGGIDSKAISSLIIKKIIETAKNNYTTLISERLNTVTVENDLLKYIKNKIPVQGRLGNFKLDTAEFEKKLNTILFVLNESNLAQRFSILVRKHYLKERPIDPIYVNILDDNSYSTLEGFNISSQGSNDFQGQLLESSYFEKIESNALRAFIKICPRNGLLYNAIYRNSKNYLNNIDLEDKKTTSKTNVSYPCSLLNGCIEVENKDLFLISNKDSLNDINLSEEKIKPETTVFFKDKLPPQDITLSNYDFTEANSIPSISQQNILERNEELYEPIRNSLFEIKTIYVDKLTTFHFLEAQNIDESIDSSKIRVELTDSVDEALSNPNKVYSPFKNMSNTINSIETGITSTYIFYQWLRSIVKDFSDETGKIDVIDKSSDTLAIVPYIGPLLNIGNDIRHGDFVGAIELAGITALLEYVPEFTIPILVGLEVIGGELAREQVEAIVNNALDKRDQKWAEVYNITKAQWWGTIHLQINTRLAHTYKALSRQANAIKMNMEFQLANYKGNIDDKAKIKNAISETEILLNKSVEQAMKNTEKFMIKLSNSYLTKEMIPKVQDNLKNFDLETKKTLDKFIKEKEDILGTNLSSSLRRKVSIRLNKNIAFDINDIPFSEFDDLINQYKNEIEDYEVLNLGAEDGKIKDLSGTTSDINIGSDIELADGRENKAIKIKGSENSTIKIAMNKYLRFSATDNFSISFWIKHPKPTNLLNNGIEYTLVENFNQRGWKISIQDSKLIWYLRDHNNSIKIVTPDYIAFNGWNLITITNNRSKGSIVYVNGSKIEEKDISSIWNTEVDDPIIFRLKNNRDTQAFTLLDQFSIYRKELNQNEVVKLYNYYFNSNYIRDIWGNPLQYNKKYYLQTQDKPGKGLIREYWSSFGYDYVILSDSKTITFPNNIRYGALYNGSKVLIKNSKKLDGLVRNKDFIQLEIDGYNMGISADRFNEDTNYIGTTYGTTHDLTTDFEIIQRQEKYRNYCQLKTPYNIFHKSGLMSTETSKPTFHDYRDWVYSSAWYFQNYENLNLRKHTKTNWYFIPKDEGWDED
""".strip().upper()

# COMMAND ----------

# TODO: should make sure nothing within X Levenshtein distance in there, too!
assert bontx not in tref50_sequences
assert bontx not in ntref50_sequences

# COMMAND ----------

sequences = ntref50_sequences + tref50_sequences
labels = tref50_labels + ntref50_labels

assert len(labels) == len(sequences)

n_labels = len(set(labels))

# COMMAND ----------

!mkdir -p /local_disk0/.hf_cache

# COMMAND ----------

# TODO: need to download these from S3, not internet!
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=n_labels)

# COMMAND ----------

train, test = train_test_split(pd.DataFrame({'sequence': sequences, 'label': labels}), test_size=0.25, shuffle=True)

# Sample down as needed
if train.shape[0] < MAX_TRAIN_SAMPLES:
    MAX_TRAIN_SAMPLES = train.shape[0]
if test.shape[0] < MAX_VALIDATE_SAMPLES:
    MAX_VALIDATE_SAMPLES = test.shape[0]
    
traindf = train.sample(n=MAX_TRAIN_SAMPLES, replace=False)
testdf = test.sample(n=MAX_VALIDATE_SAMPLES, replace=False)

# COMMAND ----------

train_fpath = os.path.join(LOCAL_TMPDIR, 'ref_train_seq.csv')
validate_fpath = os.path.join(LOCAL_TMPDIR, 'ref_test_seq.csv')

traindf.to_csv(train_fpath)
testdf.to_csv(validate_fpath)

# COMMAND ----------

print("Max sequence length allowed by tokenizer: {}".format(tokenizer.model_max_length))

print("Greatest sequeence elength input: {}".format(max([len(seq) for seq in sequences])))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Going to go on a brief tangent to create an example file that I can use to develop the BioLM ESM2 classification DAG. This will be a file that is similar to what BioLM's Django API endpoint POSTS to S3 for input to the Databricks Job.

# COMMAND ----------

spark.createDataFrame(traindf).display()

# COMMAND ----------

spark.createDataFrame(testdf).display()

# COMMAND ----------

# Stack the DFs horizontally to output a JSON file easier.
# Need to add 'source' column to output which seq is for train vs. validation.
traindf['source'] = 'train'  # TODO: Make the API accept only these two values
testdf['source'] = 'validate'
input_json_df = pd.concat([traindf, testdf], axis=0)
example_json_f = '/local_disk0/example_esm2_input.json'

payload = []
for r in input_json_df.itertuples():
    _set = r.source
    _seq = r.sequence
    _label = 'toxin' if r.label == 1 else 'benign'
    payload.append({'seq': _seq, 'label': _label, 'set': _set})

with open(example_json_f, 'w') as f:
    json.dump(payload, f)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Ok, back to the regularly scheduled program!

# COMMAND ----------

# Create Dataset for training
train_tokens = tokenizer(train_sequences)
test_tokens = tokenizer(test_sequences)

train_dataset = Dataset.from_dict(train_tokens)
test_dataset = Dataset.from_dict(test_tokens)

# COMMAND ----------

train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Define Training Loop
# MAGIC 
# MAGIC A custom callback I was thinking of modifying to help with dynamic plots as the model was trained but am still deciding whether to use it or not.

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

metric = load("accuracy")
trainingaccuracy = []

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    trainingaccuracy.append(accuracy)
    return accuracy

# COMMAND ----------

!mkdir -p /local_disk0/model_train

# COMMAND ----------

model_name = os.path.basename(MODEL_CHECKPOINT)

args = TrainingArguments(
    output_dir=os.path.join('/local_disk0/model_train'),
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    #evaluation_strategy="epoch",
    #save_strategy="epoch",
    learning_rate=2e-5,
    auto_find_batch_size=True,
    optim='adafactor',
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    #report_to=['wandb'],
    #load_best_model_at_end=True,
    #metric_for_best_model="accuracy",
    push_to_hub=False,
    save_total_limit=1
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
mlflow_exper_id = '1241984246497637'
mlflow.set_experiment(experiment_id=mlflow_exper_id)

with mlflow.start_run():
    train = trainer.train()
    train_res = trainer.evaluate()

# COMMAND ----------

train_res

# COMMAND ----------

model_out = os.path.join(LOCAL_TMPDIR, 'model_out')

# COMMAND ----------

!mkdir -p {model_out}

# COMMAND ----------

model.save_pretrained(model_out)
tokenizer.save_pretrained(model_out)

# COMMAND ----------

# Could save the trainer, too
# trainer.save_model(model_out + 'esm_fine_tuning3epochstrainer')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Load Finetuned & Predict BoNT/X

# COMMAND ----------

ft_model = AutoModelForSequenceClassification.from_pretrained(model_out, num_labels=n_labels)
ft_tokenizer = AutoTokenizer.from_pretrained(model_out)

# COMMAND ----------

# Get device
d = 0 if 'cuda' in str(device).lower() else 'cpu'

d

# COMMAND ----------

# We need tokenization and model on same device to predict
bontx_encoded = ft_tokenizer([bontx], return_tensors='pt').to(d)

with torch.no_grad():
    ft_model = ft_model.to(d)
    bontx_pred = ft_model(**bontx_encoded)
    
bontx_pred

# COMMAND ----------

bontx_proba = nn.functional.softmax(bontx_pred.logits, dim=-1)

print(bontx_proba)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Create TextClassification Pipeline

# COMMAND ----------

# I think we need to get these from the file, maybe a column named label_str.
# You can update config.json so the labels get returned by the model, too.
# TODO: save num_labels and label values in config.json, see
# https://stackoverflow.com/questions/66845379/how-to-set-the-label-names-when-using-the-huggingface-textclassificationpipeline
pipe = TextClassificationPipeline(model=ft_model, tokenizer=ft_tokenizer, device=d)

pipe(bontx)[0]
