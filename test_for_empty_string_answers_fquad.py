# from transformers import pipeline
import json
# from evaluate import load

def test_for_empty_string_or_whitespace_characters(fquad_dataset_path:str):
    fquad_dataset_file = open(fquad_dataset_path, 'r')
    fquad_dataset_string = fquad_dataset_file.read()
    fquad_dataset_file.close()

    fquad_dataset_dict = json.loads(fquad_dataset_string)
    fquad_data_list = fquad_dataset_dict["data"]

    num_empty_strings = 0
    num_whitespace_answers = 0

    for document_dict in fquad_data_list: #document_dict is a dictionary of information for each document
        paragraphs_list = document_dict["paragraphs"]
        for paragraph_dict in paragraphs_list: #this contains questions and answers from a specific context
            context = paragraph_dict["context"]
            questions_and_answers_list = paragraph_dict["qas"]
            for question_answer_dict in questions_and_answers_list:
                question = question_answer_dict["question"]
                question_id = question_answer_dict["id"]
                answer_dict = question_answer_dict["answers"][0]
                answer_text = answer_dict["text"]
                if answer_text == "":
                    num_empty_strings += 1
                if answer_text == " ":
                    num_whitespace_answers += 1
    
    print("Num empty strings:", num_empty_strings)
    print("Num whitespace answers:", num_whitespace_answers)

if __name__ == "__main__":
    fquad_train_dataset_path = "../download-form-fquad1.0/train.json"
    fquad_validation_dataset_path = "../download-form-fquad1.0/valid.json"
    test_for_empty_string_or_whitespace_characters(fquad_validation_dataset_path)
    test_for_empty_string_or_whitespace_characters(fquad_validation_dataset_path)