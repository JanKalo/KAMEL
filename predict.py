# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # , DisjunctiveConstraint
import argparse
import json
import os
import random
from tqdm import tqdm
import torch as torch

LENGTH = 100
FAST_TOKENIZATION = False
RANDOM = False
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
            few_o = "; ".join(triple['obj_label'])
            prompt += create_prompt_for_triple(few_s, p)
            prompt += " {}%\n".format(few_o)
    prompt += create_prompt_for_triple(s, p)
    return prompt


def write_prediction_file(output_path, predictions):
    with open(output_path, 'w') as f:
        for prediction in predictions:
            json.dump(prediction, f)  # JSON encode each element and write it to the file
            f.write('\n')


def predict(s, p):
    prompt = create_fewshot(s, p)
    # print("Input:")
    # print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(0)
    # get length of input
    input_length = len(inputs["input_ids"].tolist()[0])

    output = model.generate(**inputs, eos_token_id=int(tokenizer.convert_tokens_to_ids("%")),
                            max_length=input_length + LENGTH)
    generated_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    new_text = generated_text.replace(prompt, '')
    # print("Answer:")
    # print(new_text)
    # cut off the rest of the prediction.
    # Note: This only works for few shot learning with k > 0. Otherwise we might cut off things correct answers
    # pred = new_text.split('\n')[0]
    return new_text.strip()


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
    parser.add_argument('--property', help='only evaluate a single property', default='')
    parser.add_argument('--input',
                        help='Input folder path. It needs to contain subfolders with property names containing train.jsonl and test.jsonl files.')
    parser.add_argument('--fewshot', help='Number of fewshot examples', default=5, type=int)
    parser.add_argument('--templates', help='Path to template file')
    parser.add_argument('--fast', help='activates the fast tokenizer. This might not work with OPT.',
                        action='store_true')

    args = parser.parse_args()
    FAST_TOKENIZATION = args.fast

    NUMBER = args.fewshot
    model_name = args.model
    file_path = args.input
    template_path = args.templates
    if ',' in args.property:
        property = args.property.split(',')
    else:
        property = [args.property]


    print('Read parameters')
    print("Model {}\nShots {}\nProperties {}".format(model_name,NUMBER,property))
    if RANDOM == True:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=FAST_TOKENIZATION)
    print('Loaded {} model from huggingface.'.format(model_name))
    with open(template_path) as f:
        for line in f:
            p, t = line.split(',')
            templates[p] = t.replace('\n', '')
    print('Loaded template file from {}'.format(template_path))

    for subdirectory in os.listdir(file_path):
        results = []
        f = os.path.join(file_path, subdirectory)
        # checking if it is a file
        if os.path.isdir(f):
            train = read_triples(os.path.join(f, 'train.jsonl'))
            test = read_triples(os.path.join(f, 'test.jsonl'))

            #parse p from directory name
            p = str(subdirectory)

            output_path = os.path.join(f, 'predictions_{}_fewshot_{}.jsonl'.format(model_name.replace('/', ''), NUMBER))
            if os.path.isfile(output_path):
                print("Predictions for {} already exist. Skipping file.".format(str(subdirectory)))
                continue

            # if property parameter is chosen, continue until in the right subdirectory
            if property != [''] and p not in property:
                continue
            if p in templates:
                print('Evaluate examples for property {}'.format(p))
                for triple in tqdm(test):
                    prediction = predict(triple['sub_label'], p)
                    # print("correct: ", triple["obj_label"])
                    result = {'sub_label': triple['sub_label'], 'relation': p, 'obj_label': triple['obj_label'],
                              'prediction': prediction, 'fewshotk': NUMBER}
                    results.append(result)

            write_prediction_file(output_path, results)
    print('Finished evaluation')
