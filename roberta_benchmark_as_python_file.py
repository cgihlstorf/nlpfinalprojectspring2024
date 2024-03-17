# pip install datasets --quiet
# pip install evaluate --quiet

from evaluate import load
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, RobertaModel
from datasets import load_dataset
import torch, re

# model_name = "deepset/roberta-base-squad2"
model_name = "FacebookAI/roberta-base"

tokenizer = RobertaTokenizer.from_pretrained(model_name)

# model = RobertaForQuestionAnswering.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

squad_dev = load_dataset("squad_v2", split="validation")
# squad_dev = load_dataset("squad_v2", split="train+validation") # UNCOMMENT THIS WHEN YOU KNOW RESULTS CAN PRINT
squad_metric = load("squad_v2")

predictions = []
references = []

for i in range(squad_dev.num_rows):
  id = squad_dev[i]['id']
  question = squad_dev[i]["question"]
  context = squad_dev[i]["context"]
  actual_answers = squad_dev[i]["answers"]
  if len(actual_answers) == 0:
    actual_answers = ""

  tokenized_example = tokenizer(question, context, return_tensors="pt", truncation="only_second")

  with torch.no_grad():
    output = model(**tokenized_example)
    print(output)

  start_logits = output.start_logits
  end_logits = output.end_logits

  start_index = torch.argmax(start_logits)
  end_index = torch.argmax(end_logits)

  answer_tokens = tokenized_example['input_ids'][0][start_index:end_index + 1]
  predicted_answer = tokenizer.decode(answer_tokens).strip()
  # predicted_answer = re.sub(r'<s>|</s>', '', predicted_answer)
  no_answer_probability = 0
  if predicted_answer == "<s>":
    predicted_answer = ""
    no_answer_probability = 1

  predictions.append({'id': id, 'prediction_text': predicted_answer, 'no_answer_probability': no_answer_probability})
  references.append({'id': id, 'answers': actual_answers})

#   print(f"Question: {question}")
#   print(f"Actual Answer: {actual_answers}")
#   print(f"Predicted Answer: {predicted_answer}")
#   print()

results = squad_metric.compute(predictions=predictions, references=references)

print(results)
