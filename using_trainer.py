import torch
from transformers import RobertaModel, RobertaTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, Dataset

dataset_0 = load_dataset("oscar-corpus/OSCAR-2201",
                       use_auth_token=True,
                       language="fr",
                       #trust_remote_code=True,
                       streaming=True,
                       split="train"
                      )

#create a new dataset with fewer items than the original
#code from https://huggingface.co/docs/datasets/create_dataset
items_list_train = []
items_list_test = []

train_labels = []

train_counter = 0
num_train_examples = 10

test_counter = 0
num_test_examples = 5

for item in dataset_0:
  #print(train_counter, test_counter)
  if train_counter < num_train_examples and test_counter == 0:
    items_list_train.append(item)
    train_labels.append(item['meta']['identification']['label'])
    train_counter += 1
  
  elif train_counter >= num_train_examples and test_counter < num_test_examples:
    items_list_test.append(item)
    test_counter += 1

  else:
    print("Num train items:", train_counter)
    print("Num test_items:", test_counter)
    assert len(items_list_train) == num_train_examples
    assert len(items_list_test) == num_test_examples
    break
  

#all code in the next two cells from https://huggingface.co/docs/datasets/create_dataset

def gen_train():
  for item in items_list_train:
    yield item

def gen_test():
  for item in items_list_test:
    yield item

dataset_train = Dataset.from_generator(gen_train)
dataset_test = Dataset.from_generator(gen_test)

tokenizer = RobertaTokenizer.from_pretrained(base_model)

def preprocess(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    return tokenized

train_dataset = dataset_train.map(preprocess, batched=True,  remove_columns=["text"])
test_dataset = dataset_test.map(preprocess, batched=True,  remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

