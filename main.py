# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DisjunctiveConstraint
import argparse
import json
import os
import random
from tqdm import tqdm

LENGTH = 6
NUMBER = 5
test = []
train = []
templates = {}

# retrieve prompt and create single triple prompt
def create_prompt_for_triple(s, p):
    template = templates[p]
    template = template.replace('[S]', s)
    return template


# get k fewshot examples from triple list and create prompt
# this works also for number = 0

def create_fewshot(s, p):
    prompt = ""
    if NUMBER != 0:
        sample = random.sample(train, NUMBER)
        for triple in sample:
            few_s = triple['sub_label']
            few_o = triple['obj_label']
            prompt += create_prompt_for_triple(few_s, p)
            prompt += " {}\n".format(few_o)
    prompt += create_prompt_for_triple(s, p)
    return prompt


def write_prediction_file(output_path, predictions):
    with open(output_path, 'w') as f:
        for prediction in predictions:
            json.dump(prediction, f)  # JSON encode each element and write it to the file
            f.write(",\n")


def predict(s, p):
    prompt = create_fewshot(s, p)
    inputs = tokenizer(prompt, return_tensors="pt")
    # get length of input
    input_length = len(inputs["input_ids"].tolist()[0])

    output = model.generate(inputs["input_ids"].to(0), max_length=input_length + LENGTH)
    generated_text = tokenizer.decode(output[0].tolist())
    new_text = generated_text.replace(prompt, '')
    # cut off the rest of the prediction.
    # Note: This only works for few shot learning with k > 0. Otherwise we might cut off things correct answers
    pred = new_text.split('\n')[0]
    return pred.strip()


def read_triples(filepath):
    triples = []
    with open(filepath) as f:
        for line in f:
            data_instance = json.loads(line)
            triples.append(data_instance)
    return triples


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KAMEL Generative Model Predictions')
    parser.add_argument('--model', help='Put the huggingface model name here', default='gpt2-medium')
    parser.add_argument('--input', help='Input folder path. It needs to contain subfolders with property names containing train.jsonl and test.jsonl files.')
    parser.add_argument('--output', help='Output folder')
    parser.add_argument('--fewshot', help='Number of fewshot examples', default=5)
    parser.add_argument('--templates', help='Path to template file')
    args = parser.parse_args()

    NUMBER = args.fewshot
    model_name = args.model
    file_path = args.input
    output_path = args.output
    template_path = args.templates
    print('Read parameters')
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('Loaded {} model from huggingface.'.format(model_name))
    with open(template_path) as f:
        for line in f:
            p, t = line.split(',')
            templates[p] = t.replace('\n', '')
    print('Loaded template file from {}'.format(template_path))
    predictions = []
    for subdirectory in os.listdir(file_path):

        f = os.path.join(file_path, subdirectory)
        # checking if it is a file
        if os.path.isdir(f):
            train = read_triples(os.path.join(f, 'train.jsonl'))
            test = read_triples(os.path.join(f, 'test.jsonl'))
            p = str(subdirectory)
            print('Evaluate examples for property {}'.format(p))
            for triple in tqdm(test):
                prediction = predict(triple['sub_label'], p)
                predictions.append(prediction)
    print('Finished evaluation')
    write_prediction_file(output_path,predictions)
