
#TODO write a script to process the data
#TODO download camemBERT model finetuned on fquad 
#TODO do any preprocessing steps as necessaary
#TODO run camamBERT on fquad 

#NOTE it seems you structure inputs as question-context pairs

from transformers import pipeline
import json

def parse_fquad_validation_set(model, fquad_validation_dataset_path:str):
    fquad_validation_dataset_file = open(fquad_validation_dataset_path, 'r')
    fquad_validation_dataset_string = fquad_validation_dataset_file.read()
    fquad_validation_dataset_file.close()

    fquad_dataset_dict = json.loads(fquad_validation_dataset_string)
    fquad_data_list = fquad_dataset_dict["data"]

    total_num_examples = 0
    num_correct_examples = 0
    
    for document_dict in fquad_data_list: #document_dict is a dictionary of information for each document
        paragraphs_list = document_dict["paragraphs"]
        for paragraph_dict in paragraphs_list: #this contains questions and answers from a specific context
            context = paragraph_dict["context"]
            questions_and_answers_list = paragraph_dict["qas"]
            for question_answer_dict in questions_and_answers_list:
                question = question_answer_dict["question"]
                answers_list = question_answer_dict["answers"]
                #I think there should only be one dict in the list but I'm doing this just to be safe 
                for answers_dict in answers_list: 
                    true_answer = answers_dict["text"]
                    #get rid of any whitespace that may not be in a model output that is otherwise identical
                    #true_answer = true_answer.strip()
                    #don't consider case in comparing answers that may otherwise be identical except for case
                    #TODO is this model cased or uncased? Is it okay if I do this here?
                    #true_answer = true_answer.lower() 
                    
                    total_num_examples += 1

                    input_dict = ({
                        'question': question,
                        'context': context
                    })

                    #get rid of any whitespace that may make this output different from an otherwise identical true answer
                    model_answer_dict = model(input_dict)
                    model_answer = model_answer_dict['answer']
                    #get rid of any whitespace that may make this output different from an otherwise identical true answer
                    #model_answer = model_answer.strip() 
                    #don't consider case, which may be the only thing that makes the model output differ from an otherwise identical true answer
                    #model_answer = model_answer.lower()

                    #Is equality good for comparison? Should we use the BLEU score or something?
                    if model_answer == true_answer:
                        num_correct_examples += 1

    return num_correct_examples / total_num_examples

                    



#TODO they use F1 as a metric as well as exact match
#TODO I'm just using the string of the answer to measure but should I also consider the start position?

#all camembert code from https://huggingface.co/illuin/camembert-base-fquad?context=J%27etudie+l%27informatique+a+l%27universite.&text=Qu%27est-ce+que+vous+faites+pendant+votre+temps+libre%3F
fquad_validation_dataset_path = "../download-form-fquad1.0/valid.json"
model = pipeline('question-answering', model='illuin/camembert-base-fquad', tokenizer='illuin/camembert-base-fquad')
model_score = parse_fquad_validation_set(model, fquad_validation_dataset_path)
print(model_score)