
#TODO write a script to process the data
#TODO download camemBERT model finetuned on fquad 
#TODO do any preprocessing steps as necessaary
#TODO run camamBERT on fquad 

#NOTE it seems you structure inputs as question-context pairs

from transformers import pipeline
import json
from evaluate import load

def run_model_on_fquad_validation_set(model, fquad_validation_dataset_path:str):
    fquad_validation_dataset_file = open(fquad_validation_dataset_path, 'r')
    fquad_validation_dataset_string = fquad_validation_dataset_file.read()
    fquad_validation_dataset_file.close()

    fquad_dataset_dict = json.loads(fquad_validation_dataset_string)
    fquad_data_list = fquad_dataset_dict["data"]

    fquad_metric = load("squad")
    #print(fquad_metric)

    total_num_examples = 0
    num_exact_match = 0
    model_score_sum = 0
    
    references = []
    predictions = []

    for document_dict in fquad_data_list: #document_dict is a dictionary of information for each document
        paragraphs_list = document_dict["paragraphs"]
        for paragraph_dict in paragraphs_list: #this contains questions and answers from a specific context
            context = paragraph_dict["context"]
            questions_and_answers_list = paragraph_dict["qas"]
            for question_answer_dict in questions_and_answers_list:
                question = question_answer_dict["question"]
                question_id = question_answer_dict["id"]
                answer_dict = question_answer_dict["answers"][0]

                new_answer_dict = {
                    "text" : [answer_dict["text"]],
                    "answer_start": [answer_dict["answer_start"]]
                }
                
                #the ground truth
                reference = {"id": question_id, "answers": new_answer_dict}

                input_dict = ({
                    'question': question,
                    'context': context
                })

                model_output_dict = model(input_dict)
                model_answer = model_output_dict['answer']
                
                prediction = {"id": question_id, "prediction_text": model_answer, }

                references.append(reference)
                predictions.append(prediction)

    results = fquad_metric.compute(predictions=predictions, references=references)           
    print(results)

    return results

                    



#TODO they use F1 as a metric as well as exact match
#TODO I'm just using the string of the answer to measure but should I also consider the start position?

#all camembert code from https://huggingface.co/illuin/camembert-base-fquad?context=J%27etudie+l%27informatique+a+l%27universite.&text=Qu%27est-ce+que+vous+faites+pendant+votre+temps+libre%3F
fquad_validation_dataset_path = "../download-form-fquad1.0/valid.json"
model = pipeline('question-answering', model='illuin/camembert-base-fquad', tokenizer='illuin/camembert-base-fquad')
model_score = run_model_on_fquad_validation_set(model, fquad_validation_dataset_path)
print(model_score)
