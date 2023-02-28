# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import datasets
import numpy as np
import pandas as pd
from sklearn import metrics


IOU_THRESH = 0.5


# def get_questions_from_csv():
#     df = pd.read_csv("category_descriptions.csv")
#     q_dict = {}
#     for i in range(df.shape[0]):
#         category = df.iloc[i, 0].split("Category: ")[1]
#         description = df.iloc[i, 1].split("Description: ")[1]
#         q_dict[category.title()] = description
#     return q_dict


# qtype_dict = get_questions_from_csv()


def load_json(path):
    with open(path, "r") as f:
        dict = json.load(f)
    return dict


def get_preds(nbest_preds_dict, conf=None):
    results = {}
    for question_id in nbest_preds_dict:
        list_of_pred_dicts = nbest_preds_dict[question_id]
        preds = {}
        for pred_dict in list_of_pred_dicts:
            text = pred_dict["text"]
            prob = pred_dict["probability"]
            if not text == "":  # don't count empty string as a prediction
                preds[text] = prob
        preds_list = [pred for pred in preds.keys() if preds[pred] > conf]
        results[question_id] = preds_list
    return results


def get_answers(test_json_dict):
    results = {}

    data = test_json_dict["data"]
    for contract in data:
        for para in contract["paragraphs"]:
            qas = para["qas"]
            for qa in qas:
                id = qa["id"]
                answers = qa["answers"]
                answers = [answers[i]["text"] for i in range(len(answers))]
                results[id] = answers

    return results


def get_jaccard(gt, pred):
    remove_tokens = [".", ",", ";", ":"]
    for token in remove_tokens:
        gt = gt.replace(token, "")
        pred = pred.replace(token, "")
    gt = gt.lower()
    pred = pred.lower()
    gt = gt.replace("/", " ")
    pred = pred.replace("/", " ")

    gt_words = set(gt.split(" "))
    pred_words = set(pred.split(" "))

    intersection = gt_words.intersection(pred_words)
    union = gt_words.union(pred_words)
    jaccard = len(intersection) / len(union)
    return jaccard


def compute_precision_recall(gt_dict, preds_dict, category=None):
    tp, fp, fn = 0, 0, 0

    for key in gt_dict:
        if category and category not in key:
            continue

        substr_ok = "Parties" in key

        answers = gt_dict[key]
        preds = preds_dict[key]

        # first check if answers is empty
        if len(answers) == 0:
            if len(preds) > 0:
                fp += len(preds)  # false positive for each one
        else:
            for ans in answers:
                assert len(ans) > 0
                # check if there is a match
                match_found = False
                for pred in preds:
                    if substr_ok:
                        is_match = get_jaccard(ans, pred) >= IOU_THRESH or ans in pred
                    else:
                        is_match = get_jaccard(ans, pred) >= IOU_THRESH
                    if is_match:
                        match_found = True

                if match_found:
                    tp += 1
                else:
                    fn += 1

            # now also get any fps by looping through preds
            for pred in preds:
                # Check if there's a match. if so, don't count (don't want to double count based on the above)
                # but if there's no match, then this is a false positive.
                # (Note: we get the true positives in the above loop instead of this loop so that we don't double count
                # multiple predictions that are matched with the same answer.)
                match_found = False
                for ans in answers:
                    assert len(ans) > 0
                    if substr_ok:
                        is_match = get_jaccard(ans, pred) >= IOU_THRESH or ans in pred
                    else:
                        is_match = get_jaccard(ans, pred) >= IOU_THRESH
                    if is_match:
                        match_found = True

                if not match_found:
                    fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan

    return precision, recall


def process_precisions(precision):
    """
    Processes precisions to ensure that precision and recall don't both get worse
    Assumes the list precision is sorted in order of recalls
    """
    precision_best = precision[::-1]
    for i in range(1, len(precision_best)):
        precision_best[i] = max(precision_best[i - 1], precision_best[i])
    precision = precision_best[::-1]
    return precision


def get_prec_at_recall(precisions, recalls, confs, recall_thresh=0.9):
    """
    Assumes recalls are sorted in increasing order
    """
    processed_precisions = process_precisions(precisions)
    prec_at_recall = 0
    for prec, recall, conf in zip(processed_precisions, recalls, confs):
        if recall >= recall_thresh:
            prec_at_recall = prec
            break
    return prec_at_recall, conf


def get_precisions_recalls(pred_dict, gt_dict, category=None):
    precisions = [1]
    recalls = [0]
    confs = []
    for conf in list(np.arange(0.99, 0, -0.01)) + [0.001, 0]:
        conf_thresh_pred_dict = get_preds(pred_dict, conf)
        prec, recall = compute_precision_recall(
            gt_dict, conf_thresh_pred_dict, category=category
        )
        precisions.append(prec)
        recalls.append(recall)
        confs.append(conf)
    return precisions, recalls, confs


def get_aupr(precisions, recalls):
    processed_precisions = process_precisions(precisions)
    aupr = metrics.auc(recalls, processed_precisions)
    if np.isnan(aupr):
        return 0
    return aupr


def get_results(pred_dict, gt_dict, verbose=False):
    # predictions_path = os.path.join(model_path, "nbest_predictions_.json")
    # name = model_path.split("/")[-1]

    # pred_dict = load_json(predictions_path)

    assert sorted(list(pred_dict.keys())) == sorted(list(gt_dict.keys()))

    precisions, recalls, confs = get_precisions_recalls(pred_dict, gt_dict)
    prec_at_90_recall, _ = get_prec_at_recall(
        precisions, recalls, confs, recall_thresh=0.9
    )
    prec_at_80_recall, _ = get_prec_at_recall(
        precisions, recalls, confs, recall_thresh=0.8
    )
    aupr = get_aupr(precisions, recalls)

    if verbose:
        print(
            "AUPR: {:.3f}, Precision at 80% Recall: {:.3f}, Precision at 90% Recall: {:.3f}".format(
                aupr, prec_at_80_recall, prec_at_90_recall
            )
        )

    # now save results as a dataframe and return
    results = {
        # "name": name,
        "aupr": aupr,
        "prec_at_80_recall": prec_at_80_recall,
        "prec_at_90_recall": prec_at_90_recall,
    }
    return results


class CUAD(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": {
                        "id": datasets.Value("string"),
                        "prediction_text": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "probability": datasets.Value("float"),
                            }
                        ),
                    },
                    "references": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "answer_start": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
            codebase_urls=["https://www.atticusprojectai.org/cuad"],
            reference_urls=["https://www.atticusprojectai.org/cuad"],
        )

    def _compute(self, predictions, references):
        pred_dict = {}
        for prediction in predictions:
            preds = []
            for i in range(len(prediction["prediction_text"]["text"])):
                preds.append(
                    {
                        "text": prediction["prediction_text"]["text"][i],
                        "probability": prediction["prediction_text"]["probability"][i],
                    }
                )
            pred_dict[prediction["id"]] = preds
        gt_dict = {ref["id"]: ref["answers"]["text"] for ref in references}
        score = get_results(gt_dict=gt_dict, pred_dict=pred_dict)
        return score


if __name__ == "__main__":
    test_json_path = "./data/test.json"
    model_path = "./trained_models/roberta-base"
    save_dir = "./results"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    gt_dict = load_json(test_json_path)
    gt_dict = get_answers(gt_dict)

    results = get_results(model_path, gt_dict, verbose=True)

    save_path = os.path.join(save_dir, "{}.json".format(model_path.split("/")[-1]))
    with open(save_path, "w") as f:
        f.write("{}\n".format(results))
