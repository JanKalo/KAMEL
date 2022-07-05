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

LENGTH = 90
NUMBER = 5
BATCH_SIZE = 10
test = []
train = []
templates = {}
FAST_TOKENIZATION = False

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


def predict(batch, p):
    prompts = []
    for triple in batch:
        prompt = create_fewshot(triple['sub_label'], p)
        prompts.append(prompt)
    #print("Prompt lengths {}".format(len(prompts)))
    # print("Input:")
    # print(prompt)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(0)
    # get length of input
    #input_length = len(inputs["input_ids"].tolist()[0])
    #print("Input lengths {}".format(len(inputs)))
    output = model.generate(**inputs,
                                max_length=512, eos_token_id=int(tokenizer.convert_tokens_to_ids("%")))

    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)


    #generated_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    predictions = []
    for generated in generated_texts:
        new_text = generated.replace(prompt, '').strip()
        predictions.append(new_text)

    return predictions


def read_triples(filepath):
    triples = []
    with open(filepath) as f:
        for line in f:
            data_instance = json.loads(line)
            triples.append(data_instance)
    return triples


def batch(iterable, n):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KAMEL Generative Model Predictions')
    parser.add_argument('--model', help='Put the huggingface model name here', default='gpt2-medium')
    parser.add_argument('--property', help='only evaluate a single property', default='')
    parser.add_argument('--input',
                        help='Input folder path. It needs to contain subfolders with property names containing train.jsonl and test.jsonl files.')
    parser.add_argument('--fewshot', help='Number of fewshot examples', default=5, type=int)
    parser.add_argument('--batchsize', help='Batchsize', default=10, type=int)
    parser.add_argument('--templates', help='Path to template file')

    parser.add_argument('--fast', help='activates the fast tokenizer. This might not work with OPT.',  action='store_true')
    args = parser.parse_args()

    FAST_TOKENIZATION = args.fast
    NUMBER = args.fewshot
    BATCH_SIZE = args.batchsize
    model_name = args.model
    file_path = args.input
    template_path = args.templates
    property = args.property


    print('Read parameters')
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=FAST_TOKENIZATION)


    eos_token_id = int(tokenizer.convert_tokens_to_ids(".")),
    tokenizer.pad_token = tokenizer.eos_token
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
            output_path = os.path.join(f, 'predictions_{}_fewshot_{}.jsonl'.format(model_name.replace('/', ''), NUMBER))

            if os.path.isfile(output_path):
                print("Predictions for {} already exist. Skipping file.".format(str(subdirectory)))
                continue
            p = str(subdirectory)
            #if property parameter is chosen, continue until in the right subdirectory
            if property != '' and p != str(property):
                continue
            if p in templates:
                print('Evaluate examples for property {}'.format(p))
                for b in tqdm(batch(test, BATCH_SIZE)):
                    predictions = predict(b, p)
                    # print("correct: ", triple["obj_label"])
                    for triple, prediction in zip(b,predictions):
                        result = {'sub_label': triple['sub_label'], 'relation': p, 'obj_label': triple['obj_label'],
                              'prediction': prediction, 'fewshotk': NUMBER}
                        results.append(result)
            write_prediction_file(output_path, results)
    print('Finished evaluation')

