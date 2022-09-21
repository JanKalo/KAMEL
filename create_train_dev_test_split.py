import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import Counter
import glob
import random
import copy
from datetime import datetime
import os
import numpy as np
import argparse


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def get_train_dev_test_split(queries, raw_samples, train_subjects, test_subjects=None):
    train_dev_subj = random.sample(train_subjects, k=count_train_facts+count_dev_facts)
    train = random.sample(train_dev_subj, k=count_train_facts)
    dev = list(set(train_dev_subj) - set(train))
    if test_subjects:
        test = random.sample(test_subjects, k=count_test_facts)
    else:
        test = list(set(train_subjects) - set(train_dev_subj))
    #no triple overlap in train, dev and test indices
    assert len(set(train).intersection(test)) == 0
    assert len(set(train).intersection(dev)) == 0                    
    assert len(set(dev).intersection(test)) == 0 
    
    train_final = []
    for s in train:
        triples = queries[s]
        indices = []
        s_label = None
        o_labels = []
        for (i, s,s_l,p,o,o_l) in triples:
            indices.append(i)
            s_label = s_l
            o_labels.append(o_l)
        train_final.append({"index": indices, "sub_label": s_label, "obj_label": o_labels})
        
    dev_final = []
    for s in dev:
        triples = queries[s]
        indices = []
        s_label = None
        o_labels = []
        for (i, s,s_l,p,o,o_l) in triples:
            indices.append(i)
            s_label = s_l
            o_labels.append(o_l)
        dev_final.append({"index": indices, "sub_label": s_label, "obj_label": o_labels}) 
        
    test_final = []
    for s in test:
        triples = queries[s]
        indices = []
        o_uris = []
        s_label = None
        o_labels = []
        for (i, s,s_l,p,o,o_l) in triples:
            indices.append(i)
            s_label = s_l
            o_uris.append(o)
            #get whole obj_label dict
            o_l_dict = raw_samples[i]["obj_label"]
            #add obj_label to dict for which the triple was chosen as valid for given parameter
            o_l_dict["chosen"] = o_l
            o_labels.append(o_l_dict)
        test_final.append({"index": indices, "sub_label": s_label, "obj_uri": o_uris, "obj_label": o_labels})
    return train_final, dev_final, test_final

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create train, dev and test splits for KAMEL')
    parser.add_argument('--input', help='Input folder path. It needs to contain .jsonl files for each relation.')
    
    args = parser.parse_args()
    dataset = args.input
    
    label_type = ("rdf", None) # ("rdf", None) OR ("rdf", "alternative")
    count_max_answers = 10 # The maximum number of correct answers of one query
    count_train_facts = 1000
    count_dev_facts = 200
    count_test_facts = 200
    difficulty = "hard" # hard = obj label must not be in subject label OR None = all triples

    parameters = dict(dataset=dataset,
                  count_train_facts=count_train_facts,
                  count_dev_facts=count_dev_facts,
                  count_test_facts=count_test_facts,
                  label_type=label_type,
                  count_max_answers=count_max_answers
                  )

    # Get triples of given dataset
    triples = {}
    for filename in glob.glob(dataset+"*.jsonl"):
        prop = filename.split("/")[-1].replace(".jsonl", "")
        print(prop)
        triples[prop] = {}
        with open(filename, "r") as file:
            lines = list(file)
        for i, line in enumerate(lines):
            triple = json.loads(line)
            s = triple["sub_uri"]
            p = triple["predicate_id"]
            o = triple["obj_uri"]


            # get label of subject
            s_l = triple["sub_label"]["rdf"]
            #if no rdf label and label_type is only rdf, skip this triple
            if not s_l and label_type[1] != "alternative":
                continue

            #if no rdf label and label_type is alternative, choose shortest alternative label
            if not s_l and label_type[1] == "alternative":
                s_al = triple["sub_label"]["alternative"]
                s_l =  min(s_al, key=len)


            # Get label of object
            o_l = triple["obj_label"]["rdf"]

            # Choose alternative label if it is a XMLSchema (independent of the label_type)
            if "XMLSchema#" in o:
                o_al = triple["obj_label"]["alternative"]
                o_l =  min(o_al, key=len)

            # If no rdf label and label_type is only rdf, skip this triple
            if not o_l and label_type[1] != "alternative":
                continue
            # If no rdf label and label_type is alternative, choose shoortest alternative label
            if not o_l and label_type[1] == "alternative":
                o_al = triple["obj_label"]["alternative"]
                o_l =  min(o_al, key=len)

            # Check if complete object label is in subject label
            if difficulty == "hard":
                obj_in_sub = False
                s_l_words = s_l.split()
                for N in range(1, len(s_l_words)+1):
                    ngrams = [" ".join(s_l_words[j:j+N]) for j in range(len(s_l_words)-N+1)]
                    for ngram in ngrams:
                        if o_l.lower() == ngram.lower().replace(",", ""):
                            obj_in_sub = True
                            break             
            else:
                obj_in_sub = False

            if not obj_in_sub:
                # Only keep distinct (subject label, object label) pairs
                if s not in triples[prop] or not list(filter(lambda x: x[2] == s_l and x[5] == o_l, triples[prop][s])):
                    if s not in triples[prop]:
                        triples[prop][s] = []
                    triples[prop][s].append((i, s,s_l,p,o,o_l))

    # Limit number of answers of one query
    queries = copy.deepcopy(triples)
    for prop in triples:
        for s in triples[prop]:
            if len(triples[prop][s]) > count_max_answers:
                #print("WARNING query has to many answers")
                queries[prop].pop(s)

    # Make train test split of triples
    train_dev_test = {}

    for prop in queries:
        count = len(queries[prop])
        #check that prop has enough triples
        if count >= (count_train_facts+count_dev_facts+count_test_facts):
            print(f"{prop} has enough queries ({count})")
            #choose random queries of each prop
            chosen_queries = []
            while len(chosen_queries) < count_train_facts+count_dev_facts+count_test_facts:
                s = random.choice(list(queries[prop].keys()))
                queries[prop].pop(s)
                chosen_queries.append(s)
            raw_samples = load_file(f"{dataset}/{prop}.jsonl")
            train, dev, test = get_train_dev_test_split(triples[prop], raw_samples, train_subjects=chosen_queries)
            train_dev_test[prop] = dict(train=train, dev=dev, test=test)
        else:
            print(f"{prop} has not {count_train_facts+count_dev_facts+count_test_facts} queries ({count})")

    parameters["number_of_relations"] = len(train_dev_test.keys())
    now = datetime.now().strftime("%H%M%S")
    dataset_name = dataset.split("/")[-1]
    path = os.path.join("train_dev_test_splits", f"{dataset_name}_{now}")
    os.makedirs(path)
    # Save parameters
    with open(os.path.join(path, "config.json"), "w") as file:
        json.dump(parameters, file, indent=4)
    for prop in train_dev_test: 
        os.makedirs(os.path.join(path, prop))
        # Save train, dev and test split in single files
        for split in train_dev_test[prop].keys():

            with open(f"{path}/{prop}/{split}.jsonl", "w") as file:
                for triple in train_dev_test[prop][split]:
                    file.write(json.dumps(triple) + "\n")