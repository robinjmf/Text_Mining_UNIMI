#!/usr/bin/env python3

import json
import os
import os.path
import sys
import warnings
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F


def evaluate_model(model, val_dataloader, threshold):
    model.eval()
    correct = 0
    total = 0
    predictions_list = []
    labels_list = []
    with torch.no_grad():
        for batch in val_dataloader:
            if len(batch) == 3:  # Using only the first statements
                emb1_valid, emb2_valid, labels_valid = batch
                out1, out2 = model(emb1_valid, emb2_valid)
                similarity = F.cosine_similarity(out1, out2, dim=1)
                predictions = (similarity > threshold).int()

            elif len(batch) > 3:  # Using both first and second statements
                valid_samples = [(e1, e2, e3, lbl) for e1, e2, e3, lbl in zip(batch[0], batch[1], batch[2], batch[3]) if not torch.isnan(e3.clone().detach()).any()]
                if not valid_samples:
                    continue
                emb1_valid, emb2_valid, emb3_valid, labels_valid = zip(*valid_samples)
                emb1_valid = torch.stack(emb1_valid)
                emb2_valid = torch.stack(emb2_valid)
                emb3_valid = torch.stack(emb3_valid)
                labels_valid = torch.stack(labels_valid)
                if emb3_valid is not None:
                    out1, out2, out3 = model(emb1_valid, emb2_valid, emb3_valid)
                    similarity = F.cosine_similarity(out1, out3, dim=1)
                    predictions = (similarity > threshold).int()
                else:
                    out1, out2 = model(emb1_valid, emb2_valid)
                    similarity = F.cosine_similarity(out1, out2, dim=1)
                    predictions = (similarity > threshold).int()

            else:
                continue

            total += labels_valid.size(0)
            correct += (predictions == labels_valid).sum().item()
            
            # Collect predictions and labels for F1 score calculation
            predictions_list.extend(predictions.tolist())
            labels_list.extend(labels_valid.tolist())

    accuracy = correct / total
    precision = precision_score(labels_list, predictions_list)
    confusion_matrix = metrics.confusion_matrix(labels_list, predictions_list)
    recall = recall_score(labels_list, predictions_list)
    f1 = f1_score(labels_list, predictions_list)

    return accuracy, precision, recall, f1, confusion_matrix


def optimize_threshold(model, val_dataloader, threshold_values):
    accuracies = [] # Store accuracies as float numbers
    f1_scores = []
    for threshold in threshold_values:
        accuracy, precision, recall, f1, _ = evaluate_model(model, val_dataloader, threshold)
        accuracies.append(accuracy)
        f1_scores.append(f1)

    best_accuracy = max(accuracies)
    best_f1 = max(f1_scores)
    best_threshold = threshold_values[accuracies.index(best_accuracy)]
    
    plt.plot(threshold_values, accuracies)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Threshold')
    plt.show()
    
    return best_accuracy, best_f1, best_threshold


warnings.simplefilter('ignore')

def extract_control_set(predictions, gold):
    control_predicitons = {}
    for key in gold.keys():
        if "Causal_type" not in gold[key].keys():
            control_predicitons[key] = predictions[key]
    return control_predicitons


def extract_by_intervention(predictions, gold):
    para_predictions = {}
    cont_predictions = {}
    numerical_para_predictions = {}
    numerical_cont_predictions = {}
    definitions_predictions = {}
    for key in predictions.keys():
        if "Intervention" not in gold[key].keys():
            continue
        if gold[key]["Intervention"] == "Paraphrase":
            para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Contradiction":
            cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_paraphrase":
            numerical_para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_contradiction":
            numerical_cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Text_appended":
            definitions_predictions[key] = predictions[key]
    return para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions


def extract_by_causal_type(predictions, gold):
    predictions_preserving = {}
    predictions_altering = {}
    for key in predictions.keys():
        if "Causal_type" not in gold[key].keys():
            continue
        if gold[key]["Causal_type"][0] == "Preserving":
            predictions_preserving[key] = predictions[key]
        elif gold[key]["Causal_type"][0] == "Altering":
            predictions_altering[key] = predictions[key]
    return predictions_preserving, predictions_altering


def faithfulness(predictions, gold):
    uuid_list = list(predictions.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions[key]["Prediction"] != gold[gold[key]["Causal_type"][1]]["Label"]:
            results.append(1)
        else:
            results.append(0)
    Faithfulness = sum(results) / N
    return Faithfulness


def consistency(predictions, gold):
    uuid_list = list(predictions.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions[key]["Prediction"] == gold[key]["Label"]:
            results.append(1)
        else:
            results.append(0)
    Consistency = sum(results) / N
    return Consistency


def extract_contrast_set(predictions, gold):
    contrast_predicitons = {}
    for key in predictions.keys():
        if "Causal_type" in gold[key].keys():
            contrast_predicitons[key] = predictions[key]
    return contrast_predicitons


def F1_Recall_Precision(predictions, gold):
    pred_labels = []
    gold_labels = []
    for key in predictions.keys():
        if predictions[key]["Prediction"] == "Entailment":
            pred_labels.append(1)
        else:
            pred_labels.append(0)
        if gold[key]["Label"] == "Entailment":
            gold_labels.append(1)
        else:
            gold_labels.append(0)
    F1 = f1_score(gold_labels, pred_labels)
    Recall = precision_score(gold_labels, pred_labels)
    Precision = recall_score(gold_labels, pred_labels)
    return F1, Recall, Precision


def main():

    # Load files
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    pred_dir = os.path.join(input_dir, 'res')
    gold_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(pred_dir):
        raise RuntimeError('{} does not exist'.format(pred_dir))

    if not os.path.isdir(gold_dir):
        raise RuntimeError('{} does not exist'.format(gold_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gold_filename = os.path.join(gold_dir, 'gold_test.json')
    pred_filename = os.path.join(pred_dir, 'results.json')

    with open(pred_filename) as json_file:
        predictions = json.load(json_file)

    with open(gold_filename) as json_file:
        gold = json.load(json_file)


    # Control Test Set F1, Recall, Precision PUBLIC
    Control_F1, Control_Rec, Control_Prec = F1_Recall_Precision(extract_control_set(predictions, gold), gold)


    # Contrast Consistency & Faithfullness PUBLIC
    contrast_predictions = extract_contrast_set(predictions, gold)
    predictions_preserving, predictions_altering = extract_by_causal_type(contrast_predictions, gold)
    Faithfulness = faithfulness(predictions_altering, gold)
    Consistency = consistency(predictions_preserving, gold)


    # Intervention-wise Consistency & Faithfullness HIDDEN
    para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions = \
        extract_by_intervention(predictions, gold)
    para_preserving = extract_by_causal_type(para_predictions, gold)[0]
    cont_preserving, cont_altering = extract_by_causal_type(cont_predictions, gold)
    numerical_para_preserving = extract_by_causal_type(numerical_para_predictions, gold)[0]
    numerical_cont_preserving, numerical_cont_altering = extract_by_causal_type(numerical_cont_predictions, gold)
    definitions_preserving = extract_by_causal_type(definitions_predictions, gold)[0]
    para_Consistency = consistency(para_preserving, gold)
    cont_Faithfulness = faithfulness(cont_altering, gold)
    cont_Consistency = consistency(cont_preserving, gold)
    numerical_para_Consistency = consistency(numerical_para_preserving, gold)
    numerical_cont_Faithfulness = faithfulness(numerical_cont_altering, gold)
    numerical_cont_Consistency = consistency(numerical_cont_preserving, gold)
    definitions_Consistency = consistency(definitions_preserving, gold)


    # Intervention-wise F1, Recall, Precision HIDDEN
    Contrast_F1, Contrast_Rec, Contrast_Prec = F1_Recall_Precision(contrast_predictions, gold)
    para_F1, para_Rec, para_Prec = F1_Recall_Precision(para_predictions, gold)
    cont_F1, cont_Rec, cont_Prec = F1_Recall_Precision(cont_predictions, gold)
    numerical_para_F1, numerical_para_Rec, numerical_para_Prec = F1_Recall_Precision(numerical_para_predictions, gold)
    numerical_cont_F1, numerical_cont_Rec, numerical_cont_Prec = F1_Recall_Precision(numerical_cont_predictions, gold)
    definitions_F1, definitions_Rec, definitions_Prec = F1_Recall_Precision(definitions_predictions, gold)

    # Output results

    output_filename = os.path.join(output_dir, 'scores.txt')
    with open(output_filename, 'w') as f:
        print('Control_F1: ', Control_F1, file=f)
        print('Control_Recall: ', Control_Rec, file=f)
        print('Control_Precision: ', Control_Prec, file=f)
        print('Contrast_F1: ', Contrast_F1, file=f)
        print('Contrast_Recall: ', Contrast_Rec, file=f)
        print('Contrast_Precision: ', Contrast_Prec, file=f)
        print('Faithfulness: ', Faithfulness, file=f)
        print('Consistency: ', Consistency, file=f)
        print('Para_Consistency: ', para_Consistency, file=f)
        print('Cont_Faithfulness: ', cont_Faithfulness, file=f)
        print('Cont_Consistency: ', cont_Consistency, file=f)
        print('Numerical_Para_Consistency: ', numerical_para_Consistency, file=f)
        print('Numerical_Cont_Faithfulness: ', numerical_cont_Faithfulness, file=f)
        print('Numerical_Cont_Consistency: ', numerical_cont_Consistency, file=f)
        print('Definitions_Consistency: ', definitions_Consistency, file=f)
        print('Para_F1: ', para_F1, file=f)
        print('Para_Recall: ', para_Rec, file=f)
        print('Para_Precision: ', para_Prec, file=f)
        print('Cont_F1: ', cont_F1, file=f)
        print('Cont_Recall: ', cont_Rec, file=f)
        print('Cont_Precision: ', cont_Prec, file=f)
        print('Numerical_Para_F1: ', numerical_para_F1, file=f)
        print('Numerical_Para_Recall: ', numerical_para_Rec, file=f)
        print('Numerical_Para_Precision: ', numerical_para_Prec, file=f)
        print('Numerical_Cont_F1: ', numerical_cont_F1, file=f)
        print('Numerical_Cont_Recall: ', numerical_cont_Rec, file=f)
        print('Numerical_Cont_Precision: ', numerical_cont_Prec, file=f)
        print('Definitions_F1: ', definitions_F1, file=f)
        print('Definitions_Recall: ', definitions_Rec, file=f)
        print('Definitions_Precision: ', definitions_Prec, file=f)


if '__main__' == __name__:
    main()










