from datetime import datetime
import datasets
from evaluate import load
import json
import numpy as np
import os
import requests
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from sentence_transformers import InputExample, models, SentenceTransformer, models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import DenoisingAutoEncoderLoss
import sys
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW
from transformers import RobertaForQuestionAnswering, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import re
from datasets import load_dataset

def preprocess_squad2_for_sentence_transformers(dataset, output_file_path:str, create_new_file:bool = True):

    items_to_predict = []
    reference_points = []

    for item in dataset:
        item_id = item["id"]
        context = item["context"]
        question = item["question"]
        answers_original = item["answers"] #make this into a list
        # print(answers_original)
        # print("===========================")
        # print(item)
        # answers = {
        #     "text": answers_original["text"][0],
        #     "answer_start": answers_original["answer_start"][0]
        # }
        # print(answers)
        # print(item)
        # break
        
        reference_point = {
                   "id": item_id,
                   "answers": answers_original,
                }
            
        reference_points.append(reference_point)

        item_to_predict = {
            "context": context,
            "qas": [
                {
                    "question": question,
                    "id": item_id,
                }
            ]
        }

        items_to_predict.append(item_to_predict)

    # if create_new_file == True:
    #     output_file = open(output_file_path, 'w')
    #     eval_data_json_string = json.dumps(eval_data_list, indent=1)
    #     output_file.write(eval_data_json_string)
    #     output_file.close()

    return items_to_predict, reference_points

def benchmark_roberta_base_on_squad_v2(new_squad_2_predict_data, new_squad_2_eval_data):

    print(torch.cuda.is_available())

    #model = QuestionAnsweringModel("sentence-transformers/stsb-roberta-base-v2")
    model = QuestionAnsweringModel("roberta",
                               "sentence-transformers/stsb-roberta-base-v2",
                               #args=train_args,
                               use_cuda=False)

    squad_v2_metric = load("squad_v2")

    model_predictions, raw_outputs = model.predict(new_squad_2_predict_data)

    #print(model_predictions)

    question_ind = {}
    predictions = []

    print("got here 1")
    for raw_output in raw_outputs:
        question_id = raw_output['id']
        probabilities = raw_output['probability']
        ind_w_max = np.argmax(probabilities)
        question_ind[question_id] = ind_w_max

    print("got here 2")
    for model_prediction in model_predictions:
        question_id = model_prediction['id']
        ind_of_ans = question_ind[question_id]
        answer = model_prediction['answer'][ind_of_ans]
        prediction = {"id": question_id, "prediction_text": answer, "no_answer_probability": 0.0}
        predictions.append(prediction)

    print("got here 3")
    #print(predictions[0])

    results = squad_v2_metric.compute(predictions=predictions, references=new_squad_2_eval_data)           
    print(results)

# Sentence Transformers:

# eval_data = [
#     {
#         "context": "The series primarily takes place in a region called the Final Empire "
#                    "on a world called Scadrial, where the sun and sky are red, vegetation is brown, "
#                    "and the ground is constantly being covered under black volcanic ashfalls.",
#         "qas": [
#             {
#                 "id": "00001",
#                 "is_impossible": False,
#                 "question": "Where does the series take place?",
#                 "answers": [
#                     {
#                         "text": "region called the Final Empire",
#                         "answer_start": 38,
#                     },
#                     {
#                         "text": "world called Scadrial",
#                         "answer_start": 74,
#                     },
#                 ],
#             }
#         ],
#     },
# ]


# Squad V2:
# {
# 'id': '56be85543aeaaa14008c9063', 
# 'title': 'Beyoncé', 
# 'context': 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".', 
# 'question': 'When did Beyonce start becoming popular?', 
# 'answers': 
#     {
#         'text': ['in the late 1990s'],
#         'answer_start': [269]
#         }
# }


squad_2_dataset_val = load_dataset("rajpurkar/squad_v2", streaming=True)["validation"]
output_file_path_val = "preprocessed_squad2_val.json"

squad_v2_items_to_predict, reference_points = preprocess_squad2_for_sentence_transformers(squad_2_dataset_val, output_file_path_val, create_new_file=False)
print("Got to model predictions!")
# print(items_to_predict[0:2])
benchmark_roberta_base_on_squad_v2(squad_v2_items_to_predict, reference_points)