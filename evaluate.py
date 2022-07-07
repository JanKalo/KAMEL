import json
import os
import tqdm
import argparse


def read_triples(filepath):
    triples = []
    with open(filepath) as f:
        for line in f:
            # line = line[:-2]
            data_instance = json.loads(line)
            triples.append(data_instance)
    return triples


all_prec = 0
all_rec = 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KAMEL Generative Model Predictions')
    parser.add_argument('--model', help='Put the huggingface model name here', default='gpt2-medium')
    parser.add_argument('--input',
                        help='Input folder path. It needs to contain subfolders with property names containing train.jsonl and test.jsonl files.')

    args = parser.parse_args()

    for subdirectory in os.listdir(args.input):
        results = []
        f = os.path.join(args.input, subdirectory)
        # checking if it is a file
        if os.path.isdir(f):
            predictions = read_triples(os.path.join(f, args.model))
            avg_prec = 0.0
            avg_rec = 0.0
            for triple in predictions:
                prediction = triple['prediction'].replace('%', '')
                # print(prediction)

                gold_answers = []
                if isinstance(triple['obj_label'], str):
                    p = set()
                    p.add(prediction)
                    gold_set = set()
                    gold_set.add(triple['obj_label'])
                    gold_answers.append(gold_set)
                else:
                    p = set(x.lower() for x in prediction.split(';'))
                    for g in triple['obj_label']:
                        gold_set = set()
                        gold_set.add(g['chosen'].lower())
                        for x in g['alternative']:
                            gold_set.add(x.lower())
                        gold_answers.append(gold_set)
                try:
                    no_correct = 0
                    for gold_set in gold_answers:
                        if len(gold_set & p) != 0:
                            no_correct += 1
                    precision = (no_correct / (len(p)))
                    recall = (no_correct / len(gold_answers))
                except ZeroDivisionError:
                    print(prediction)
                    continue
                avg_prec += precision
                avg_rec += recall
            try:
                avg_prec = (avg_prec / len(predictions))
                avg_rec = (avg_rec / len(predictions))
                print(subdirectory, avg_prec, avg_rec)
            except ZeroDivisionError:
                continue
            all_prec += avg_prec
            all_rec += avg_rec

    # TO DO: Die Nenner müssen angepasst werden.
    print((all_prec / 240.0), (all_rec / 240.0))
