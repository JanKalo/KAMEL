import argparse
import json
import glob
from operator import index
import pandas as pd

MODEL_NAMES = {"EleutherAIgpt-j-6B": "GPT-J-6b",
                "bigsciencebloom-1b3": "Bloom-1b3",
                "facebookopt-13b": "OPT-13b",
                "facebookopt-6.7b": "OPT-6.7b",
                "facebookopt-1.3b": "OPT-1.3b",
                "gpt2-xl": "GPT2-xl",
                "gpt2-medium": "GPT2-medium"}

#except P530 because for our dataset P530 has not enough queries for train-dev-test
LAMA_RELS = {"P1001", "P101", "P103", "P106", "P108", "P127", "P1303", "P131", "P136", "P1376", "P138", "P140", "P1412", "P159", "P17", "P176", "P178", "P19", "P190", "P20", "P264", "P27", "P276", "P279", "P30", "P31", "P36", "P361", "P364", "P37", "P39", "P407", "P413", "P449", "P463", "P47", "P495", "P527", "P740", "P937"}

REMOVED_RELS = {}

F1 = lambda prec, rec: 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0 

to_percent = lambda x: '{:.2%}'.format(x)

def read_triples(filepath):
    triples = []
    with open(filepath) as f:
        for line in f:
            data_instance = json.loads(line)
            triples.append(data_instance)
    return triples

def get_meta_info(predictions_path):
    relation = predictions_path.split("/")[-2]
    model_name = MODEL_NAMES[predictions_path.split("/")[-1].split("_")[1]]
    number = predictions_path.split("/")[-1].split("_")[-1].replace(".jsonl", "")
    return relation, model_name, number

def evaluate():
    results_per_prop = {}
    for predictions_path in glob.glob(FILE_PATH + "/*/predictions_*jsonl"):
        predictions = read_triples(predictions_path)
        if len(predictions) == 0:
            continue

        relation, model_name, number = get_meta_info(predictions_path)
        if model_name ==  "GPT2-medium" or model_name ==  "Bloom-1b3":
            continue 
        if model_name not in results_per_prop:
            results_per_prop[model_name] = {}
        if number not in results_per_prop[model_name]:
            results_per_prop[model_name][number] = {}
        if relation not in results_per_prop[model_name][number]:
            results_per_prop[model_name][number][relation] = {}
        
        #prec and recall when only using rdf labels
        avg_prec_rdf = 0.0
        avg_rec_rdf = 0.0
        #prec and recall when using rdf AND alternative labels
        avg_prec_altlabel = 0.0
        avg_rec_altlabel = 0.0
        for triple in predictions:
            pred = triple['prediction'].replace('%','')
            
            if pred:
                pred = set(x.lower() for x in pred.split(';'))
            else:
                pred = set()
            gold_answers = []
            for g in triple['obj_label']:
                gold_set = {}
                gold_set["rdf"] = {g['rdf'].lower()} if g['rdf'] else set()
                gold_set["rdf_altlabel"] = gold_set["rdf"].copy()
                for x in g['alternative']:
                    if not gold_set["rdf"]:
                        gold_set["rdf"].add(x.lower())
                    gold_set["rdf_altlabel"].add(x.lower())
                gold_answers.append(gold_set)
            try:
                correct_rdf = 0
                correct_altlabel = 0
                for gold_set in gold_answers:
                    if len(gold_set["rdf"]&pred) != 0:
                        correct_rdf += 1
                    if len(gold_set["rdf_altlabel"]&pred) != 0:
                        correct_altlabel += 1
                precision_rdf = (correct_rdf/(len(pred)))
                recall_rdf = (correct_rdf/len(gold_answers))
                precision_altlabel = (correct_altlabel/(len(pred)))
                recall_altlabel = (correct_altlabel/len(gold_answers))
            except ZeroDivisionError:
                continue
            avg_prec_rdf += precision_rdf
            avg_rec_rdf += recall_rdf
            avg_prec_altlabel += precision_altlabel
            avg_rec_altlabel += recall_altlabel
        
        avg_prec_rdf = (avg_prec_rdf/len(predictions))
        avg_rec_rdf = (avg_rec_rdf/len(predictions))
        avg_prec_altlabel = (avg_prec_altlabel/len(predictions))
        avg_rec_altlabel = (avg_rec_altlabel/len(predictions))
        results_per_prop[model_name][number][relation]["avg_prec"] = {"rdf": avg_prec_rdf, "rdf + alternative": avg_prec_altlabel}
        results_per_prop[model_name][number][relation]["avg_rec"] = {"rdf": avg_rec_rdf, "rdf + alternative": avg_rec_altlabel}
    
    results = {}
    for model_name in results_per_prop:
        results[model_name] = {}
        for number in results_per_prop[model_name]:
            results[model_name][number] = {}
            all_prec_rdf = 0
            all_rec_rdf = 0
            all_prec_altlabel = 0
            all_rec_altlabel = 0
            for relation in results_per_prop[model_name][number]:
                all_prec_rdf += results_per_prop[model_name][number][relation]["avg_prec"]["rdf"]
                all_rec_rdf += results_per_prop[model_name][number][relation]["avg_rec"]["rdf"]
                all_prec_altlabel += results_per_prop[model_name][number][relation]["avg_prec"]["rdf + alternative"]
                all_rec_altlabel += results_per_prop[model_name][number][relation]["avg_rec"]["rdf + alternative"]

                results_per_prop[model_name][number][relation]["avg_prec"] = results_per_prop[model_name][number][relation]["avg_prec"]["rdf"]
                results_per_prop[model_name][number][relation]["avg_rec"] = results_per_prop[model_name][number][relation]["avg_rec"]["rdf"]
                results_per_prop[model_name][number][relation]["F1"] = F1(results_per_prop[model_name][number][relation]["avg_prec"], results_per_prop[model_name][number][relation]["avg_rec"])
            all_prec_rdf = all_prec_rdf/len(results_per_prop[model_name][number].keys())
            all_rec_rdf = all_rec_rdf/len(results_per_prop[model_name][number].keys())
            all_prec_altlabel = all_prec_altlabel/len(results_per_prop[model_name][number].keys())
            all_rec_altlabel = all_rec_altlabel/len(results_per_prop[model_name][number].keys())
            results[model_name][number]["all_prec"] = f"{to_percent(all_prec_altlabel)} ({to_percent(all_prec_rdf)})"
            results[model_name][number]["all_rec"] = f"{to_percent(all_rec_altlabel)} ({to_percent(all_rec_rdf)})"
            results[model_name][number]["F1"] = f"{to_percent(F1(all_prec_altlabel, all_rec_altlabel))} ({to_percent(F1(all_prec_rdf, all_rec_rdf))})"
    return results, results_per_prop


def evaluate_cardinality():
    results_cardinality = {}
    for predictions_path in glob.glob(FILE_PATH + f"/*/predictions_*_fewshot_{NUMBER}.jsonl"):
        predictions = read_triples(predictions_path)
        if len(predictions) == 0:
            continue
        _, model_name, _ = get_meta_info(predictions_path)
        if model_name ==  "GPT2-medium" or model_name ==  "Bloom-1b3":
            continue
        if model_name not in results_cardinality:
            results_cardinality[model_name] = {}
        
        for triple in predictions:
            pred = triple['prediction'].replace('%','')
            if pred:
                pred = set(x.lower() for x in pred.split(';'))
            else:
                pred = set()
            gold_answers = []
            for g in triple['obj_label']:
                gold_set = set()
                if g['rdf']:
                    gold_set.add(g['rdf'].lower())
                for x in g['alternative']:
                    gold_set.add(x.lower())
                gold_answers.append(gold_set)
            try:
                if len(gold_answers) not in results_cardinality[model_name]:
                    results_cardinality[model_name][len(gold_answers)] = {}
                    results_cardinality[model_name][len(gold_answers)]["count"] = 0
                    results_cardinality[model_name][len(gold_answers)]["all_prec"] = 0
                    results_cardinality[model_name][len(gold_answers)]["all_rec"] = 0
                results_cardinality[model_name][len(gold_answers)]["count"] += 1
                correct = 0
                for gold_set in gold_answers:
                    if len(gold_set&pred) != 0:
                        correct += 1
                
                precision = (correct/(len(pred)))
                recall = (correct/len(gold_answers))
            except ZeroDivisionError:
                continue
            results_cardinality[model_name][len(gold_answers)]["all_prec"] += precision
            results_cardinality[model_name][len(gold_answers)]["all_rec"] += recall
           
    for model_name in results_cardinality:
        for count_gold_answers in results_cardinality[model_name]:
            all_prec = (results_cardinality[model_name][count_gold_answers]["all_prec"]/results_cardinality[model_name][count_gold_answers]["count"])
            all_rec = (results_cardinality[model_name][count_gold_answers]["all_rec"]/results_cardinality[model_name][count_gold_answers]["count"])
            results_cardinality[model_name][count_gold_answers]["all_prec"] = all_prec
            results_cardinality[model_name][count_gold_answers]["all_rec"] = all_rec
            results_cardinality[model_name][count_gold_answers]["F1"] = F1(all_prec, all_rec)
    return results_cardinality

def evaluate_query_type():
    results_query_type = {}
    for predictions_path in glob.glob(FILE_PATH + f"/*/predictions_*_fewshot_{NUMBER}.jsonl"):
        predictions = read_triples(predictions_path)
        if len(predictions) == 0:
            continue
        relation, model_name, _ = get_meta_info(predictions_path)
        if model_name ==  "GPT2-medium" or model_name ==  "Bloom-1b3":
            continue
        if model_name not in results_query_type:
            results_query_type[model_name] = {}

        for triple in predictions:
            pred = triple['prediction'].replace('%','')
            if pred:
                pred = set(x.lower() for x in pred.split(';'))
            else:
                pred = set()
            gold_answers = []
            #flag to differentiate between a normal entity or a number entity (e.g. mass, date, year)
            is_number = False
            for g in triple['obj_label']:
                gold_set = set()
                if g['rdf']:
                    gold_set.add(g['rdf'].lower())
                else:
                    is_number = True
                for x in g['alternative']:
                    gold_set.add(x.lower())

                gold_answers.append(gold_set)
            try:
                if not is_number:
                    key = "Entites"
                else:
                    key = "Literals"
                if key not in results_query_type[model_name]:
                    results_query_type[model_name][key] = {}
                    results_query_type[model_name][key]["count"] = 0
                    results_query_type[model_name][key]["all_prec"] = 0
                    results_query_type[model_name][key]["all_rec"] = 0
                results_query_type[model_name][key]["count"] += 1
                correct = 0
                for gold_set in gold_answers:
                    if len(gold_set&pred) != 0:
                        correct += 1
                
                precision = (correct/(len(pred)))
                recall = (correct/len(gold_answers))
            except ZeroDivisionError:
                continue
            results_query_type[model_name][key]["all_prec"] += precision
            results_query_type[model_name][key]["all_rec"] += recall
    for model_name in results_query_type:
        for query_type in results_query_type[model_name]:
            all_prec = (results_query_type[model_name][query_type]["all_prec"]/results_query_type[model_name][query_type]["count"])
            all_rec = (results_query_type[model_name][query_type]["all_rec"]/results_query_type[model_name][query_type]["count"])
            results_query_type[model_name][query_type]["all_prec"] = all_prec
            results_query_type[model_name][query_type]["all_rec"] = all_rec
            results_query_type[model_name][query_type]["F1"] = F1(all_prec, all_rec)
    return results_query_type

def evaluate_LAMA_rels():
    results_LAMA_rels_per_prop = {}
    for i, path in enumerate([FILE_PATH, FILE_PATH_LAMA]):
        if i==0:
            dataset = "KAMEL"
        elif i == 1:
            dataset = "LAMA"
        results_LAMA_rels_per_prop[dataset] = {}

        for predictions_path in glob.glob(path + f"/*/predictions_facebookopt-13b_fewshot_*.jsonl"):
            predictions = read_triples(predictions_path)
            if len(predictions) == 0:
                continue
            relation, _, number = get_meta_info(predictions_path)
            if relation not in LAMA_RELS:
                continue
            if number not in results_LAMA_rels_per_prop[dataset]:
                results_LAMA_rels_per_prop[dataset][number] = {}
            if relation not in results_LAMA_rels_per_prop[dataset][number]:
                results_LAMA_rels_per_prop[dataset][number][relation] = {}
            
            avg_prec = 0.0
            avg_rec = 0.0

            for triple in predictions:
                pred = triple['prediction'].replace('%','')
                if pred:
                    pred = set(x.lower() for x in pred.split(';'))
                else:
                    pred = set()
                
                gold_answers = []
                if isinstance(triple['obj_label'], list):
                    for g in triple['obj_label']:
                        gold_set = set()
                        if g['rdf']:
                            gold_set.add(g['rdf'].lower())
                        for x in g['alternative']:
                            gold_set.add(x.lower())

                        gold_answers.append(gold_set)
                else:
                    gold_answers.append({triple['obj_label'].lower()})
                try:
                    correct = 0
                    for gold_set in gold_answers:
                        if len(gold_set&pred) != 0:
                            correct += 1
                    precision = (correct/(len(pred)))
                    recall = (correct/len(gold_answers))
                except ZeroDivisionError:
                    #print(triple)
                    continue
                avg_prec += precision
                avg_rec += recall
            avg_prec = (avg_prec/len(predictions))
            avg_rec = (avg_rec/len(predictions))
            results_LAMA_rels_per_prop[dataset][number][relation]["avg_prec"] = avg_prec
            results_LAMA_rels_per_prop[dataset][number][relation]["avg_rec"] = avg_rec
    
    results_LAMA_rels = {}
    for dataset in results_LAMA_rels_per_prop:
        results_LAMA_rels[dataset] = {}
        for number in results_LAMA_rels_per_prop[dataset]:
            results_LAMA_rels[dataset][number] = {}
            all_prec = 0
            all_rec = 0
            for relation in results_LAMA_rels_per_prop[dataset][number]:
                all_prec += results_LAMA_rels_per_prop[dataset][number][relation]["avg_prec"]
                all_rec += results_LAMA_rels_per_prop[dataset][number][relation]["avg_rec"]
            assert len(results_LAMA_rels_per_prop[dataset][number].keys()) == len(LAMA_RELS)    
            all_prec = all_prec/len(LAMA_RELS)  
            all_rec = all_rec/len(LAMA_RELS)  
            results_LAMA_rels[dataset][number]["all_prec"] = all_prec
            results_LAMA_rels[dataset][number]["all_rec"] = all_rec
            results_LAMA_rels[dataset][number]["F1"] = F1(all_prec, all_rec)
    return results_LAMA_rels, results_LAMA_rels_per_prop
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creation of result tables and entity type/cardinality plot')
    parser.add_argument('--input', help='Input folder path of our new dataset. It needs to contain subfolders with property names containing train.jsonl and test.jsonl files.')
    parser.add_argument('--input_LAMA', help='Input folder path of Autorprompt and LAMA dataset. It needs to contain subfolders with property names containing train.jsonl and test.jsonl files.')
    #These parameters are only for the cardinality plot
    parser.add_argument('--fewshot', help='Number of fewshot examples', default=10, type=int)
    args = parser.parse_args()

    FILE_PATH = args.input
    FILE_PATH_LAMA = args.input_LAMA
    NUMBER = args.fewshot

    results, results_per_prop = evaluate()
    df = pd.DataFrame.from_dict(results, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.index.names = ['Model', "Fewshot"]
    #df = df.reset_index()  
    #df.Fewshot = df.Fewshot.astype(int)
    df.columns = ['Prec (%)', 'Rec (%)', 'F1 (%)']
    with open('results.tex', 'w+') as tf:
        tf.write(df.to_latex())
    print(df)

    results_LAMA_rels, results_LAMA_rels_per_prop = evaluate_LAMA_rels()
    df = pd.DataFrame.from_dict(results_LAMA_rels, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.loc[:, "all_prec"] = df["all_prec"].map('{:.2%}'.format)
    df.loc[:, "all_rec"] = df["all_rec"].map('{:.2%}'.format)
    df.loc[:, "F1"] = df["F1"].map('{:.2%}'.format)
    df.columns = ['P', 'R', 'F1']
    with open('results_LAMA.tex', 'w+') as tf:
        tf.write(df.to_latex())
    print(df)

    results_cardinality = evaluate_cardinality()
    df = pd.DataFrame.from_dict(results_cardinality, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.index.names = ['Model', "Cardinality"] 
    df.loc[:, "F1"] = df["F1"].map('{:.2%}'.format)
    df["F1"] = df["F1"].str.rstrip("%").astype(float)
    print(df)
    df = df.reset_index() 
    df_pv = df.pivot(index='Cardinality', columns='Model', values='F1')
    plot = df_pv.plot(kind="line", rot=0,  xticks=df.Cardinality, yticks=range(0, 27, 5), ylabel="F1 [%]")
    fig = plot.get_figure()
    fig.savefig("output_cardinality.png")

    results_query_type = evaluate_query_type()
    df = pd.DataFrame.from_dict(results_query_type, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.index.names = ['Model', "Query Type"]
    df.loc[:, "F1"] = df["F1"].map('{:.2%}'.format)
    df["F1"] = df["F1"].str.rstrip("%").astype(float)
    print(df)
    df = df.reset_index()  
    df = df.pivot(index='Model', columns='Query Type', values='F1')
    plot = df.plot(kind="bar", rot=0, yticks=range(0, 27, 5), ylabel="F1 [%]")
    fig = plot.get_figure()
    fig.savefig("output_query_type.png")

    
    

