import os, sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import argparse

import torch
import pickle
import json
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from preprocessing.codebert.CodebertDataset import CodebertDataset
from preprocessing.Defects4JLoader import Defects4JLoader
from preprocessing.pmt_baseline.PMTDataset import PMTDataset
from preprocessing.pmt_baseline.PMTDataLoader import PMTDataLoader
from Macros import Macros
from preprocessing import utils
from pathlib import Path
from tqdm import tqdm
import numpy as np

MODEL_CONFIG_DICT = {
    "pmt_baseline": {
        "type": "pmt_baseline",
        "data_name": "pmt_baseline_ordered",
        "best_checkpoint": "seshat_same_project_matrix.pth.tar",
    },
    "pmt_baseline_cp": {
        "type": "pmt_baseline",
        "data_name": "pmt_baseline_ordered_cp",
        "best_checkpoint": "seshat_cross_project_matrix.pth.tar",
    },
    "codebert_token_diff": {
        "type": "unary",
        "data_name": "codebert_token_diff",
        "best_checkpoint": "mutationbert_same_project_matrix.pth.tar",
    },
    "codebert_token_diff_cp": {
        "type": "unary",
        "data_name": "codebert_token_diff_cp",
        "best_checkpoint": "mutationbert_cross_project_matrix.pth.tar",
    },
}

def compute_metrics(preds, labels):
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    return precision, recall, f1, accuracy

def eval_dataset_pmt(model, device, dataloader):
    model.eval()
    with torch.no_grad():
        labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
        with tqdm(dataloader, unit="batch") as tepoch:
            for _, sents1, sents2, body, before, after, mutator, labels in tepoch:
                sents1 = sents1.to(device)
                sents2 = sents2.to(device)
                body = body.to(device)
                before = before.to(device)
                after = after.to(device)
                mutator = mutator.to(device)
                labels = labels.to(device)

                labels_data = torch.cat([labels_data, labels.cpu()])

                scores, _, _, _ = model(sents1, sents2, body, before, after, mutator)
                predictions = scores.max(dim=1)[1]
                preds_data = torch.cat([preds_data, predictions.cpu()])

    return labels_data, preds_data


def eval_dataset_unary(model, device, dataloader):
    model.eval()
    with torch.no_grad():
        labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch_idx, (ids, mask, idx, labels) in enumerate(tepoch):
                ids = ids.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                idx = idx.to(device)

                labels_data = torch.cat([labels_data, labels.cpu()])

                scores = model(ids, mask)
                predictions = scores.max(dim=1)[1]
                preds_data = torch.cat([preds_data, predictions.cpu()])
    
    return labels_data, preds_data

def compute_unary_stats(model_dir, device, model_details):
    validation_dataset = CodebertDataset(Macros.data_dir / model_details["data_name"] / "val")
    validation_dataloader = Defects4JLoader(validation_dataset, 1)
    test_dataset = CodebertDataset(Macros.data_dir / model_details["data_name"] / "test")
    test_dataloader = Defects4JLoader(test_dataset, 1)

    model_dict = torch.load(model_dir / model_details["best_checkpoint"])
    model = model_dict["model"].to(device)

    val_labels, val_preds = eval_dataset_unary(model, device, validation_dataloader)
    test_labels, test_preds = eval_dataset_unary(model, device, test_dataloader)

    val_precision, val_recall, val_f1, val_accuracy = compute_metrics(val_preds, val_labels)
    test_precision, test_recall, test_f1, test_accuracy = compute_metrics(val_preds, val_labels)

    return {"val_prec": val_precision, "val_recall": val_recall, "val_f1": val_f1, "val_accuracy": val_accuracy, "test_prec": test_precision, "test_recall": test_recall, "test_f1": test_f1, "test_accuracy": test_accuracy}

def compute_pmt_stats(model_dir, device, model_details):
    base_path = Macros.data_dir / model_details["data_name"]
    validation_dataset = PMTDataset(base_path / "val",
                                    base_path / "vocab_method_name.pkl", base_path / "vocab_body.pkl", max_sent_length=150)
    validation_dataloader = PMTDataLoader(validation_dataset, 512)
    
    test_dataset = PMTDataset(base_path / "test",
                                base_path / "vocab_method_name.pkl", base_path / "vocab_body.pkl", max_sent_length=150)
    test_dataloader = PMTDataLoader(test_dataset, 512)

    model_dict = torch.load(model_dir / model_details["best_checkpoint"])
    model = model_dict["model"].to(device)

    val_labels, val_preds = eval_dataset_pmt(model, device, validation_dataloader)
    test_labels, test_preds = eval_dataset_pmt(model, device, test_dataloader)

    val_precision, val_recall, val_f1, val_accuracy = compute_metrics(val_preds, val_labels)
    test_precision, test_recall, test_f1, test_accuracy = compute_metrics(test_preds, test_labels)

    return {"val_prec": val_precision, "val_recall": val_recall, "val_f1": val_f1, "val_accuracy": val_accuracy, "test_prec": test_precision, "test_recall": test_recall, "test_f1": test_f1, "test_accuracy": test_accuracy}

def build_model_datapoints(data_dir):
    data_pts = {}
    i = 0
    for file_ind, path in enumerate(os.listdir(data_dir)):
        with open(os.path.join(data_dir, path), "rb") as f:
            mutants = pickle.load(f)
            for mutant in mutants:
                actual_ind = file_ind * 10_000 + mutant["id"]
                if actual_ind not in data_pts:
                    data_pts[actual_ind] = {}
                data_pts[actual_ind][mutant["type"]] = (torch.LongTensor(mutant["embed"]), torch.LongTensor(mutant["mask"]), torch.LongTensor([mutant["index"]]), mutant["label"])
    return data_pts

def get_binary_preds(model, device, datapts):
    model.eval()
    preds_map = {}
    labels_map = {}
    with torch.no_grad():
        labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
        for idx in tqdm(datapts):
            preds_map[idx] = {}
            labels_map[idx] = {}
            ids, mask, _, labels = datapts[idx]["orig"]
            mut_ids, mut_mask, _, mut_labels = datapts[idx]["mutated"]

            ids = ids.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            labels = torch.LongTensor([labels]).to(device)
            
            mut_ids = mut_ids.unsqueeze(0).to(device)
            mut_mask = mut_mask.unsqueeze(0).to(device)
            mut_labels = torch.LongTensor([mut_labels]).to(device)

            labels_map[idx]["orig"] = labels[0].cpu()
            labels_map[idx]["mutated"] = mut_labels[0].cpu()

            scores = model(ids, mask)
            scores_mutated = model(mut_ids, mut_mask)

            preds_map[idx]["orig"] = scores[0][1].cpu()
            preds_map[idx]["mutated"] = scores_mutated[0][1].cpu()
    
    return labels_map, preds_map

def compute_threshold_metrics(labels_map, preds_map, threshold):
    labels, preds = [], []
    for idx in labels_map:
        pred = 1 if (preds_map[idx]["mutated"] - preds_map[idx]["orig"]) >= threshold else 0
        preds.append(pred)
        labels.append(labels_map[idx]["mutated"])
    
    return preds, labels

def compute_binary_metrics(labels_map, preds_map, final_threshold=None):
    MIN_THRESHOLD = 0.01
    MAX_THRESHOLD = 1.0
    STEP = 0.01

    metric_map = {"normal": {}, "threshold": {}}

    labels_norm, preds_norm = [], []
    for idx in labels_map:
        pred = 1 if preds_map[idx]["mutated"] >= 0.5 else 0
        preds_norm.append(pred)
        labels_norm.append(labels_map[idx]["mutated"])
    
    prec, recall, f1, acc = compute_metrics(preds_norm, labels_norm)
    metric_map["normal"] = {"prec": prec, "recall": recall, "f1": f1, "acc": acc}

    if final_threshold is None:
        max_f1 = 0
        final_threshold = 0
        for threshold in np.arange(MIN_THRESHOLD, MAX_THRESHOLD, STEP):
            preds, labels = compute_threshold_metrics(labels_map, preds_map, threshold)
            curr_prec, curr_recall, curr_f1, curr_acc = compute_metrics(preds, labels)
            if curr_f1 > max_f1:
                max_f1 = curr_f1
                final_threshold = threshold
    
    preds, labels = compute_threshold_metrics(labels_map, preds_map, final_threshold)
    curr_prec, curr_recall, curr_f1, curr_acc = compute_metrics(preds, labels)
    metric_map["threshold"] = {"prec": curr_prec, "recall": curr_recall, "f1": curr_f1, "acc": curr_acc, "threshold": final_threshold}
    print(metric_map)
    return metric_map, final_threshold

def compute_binary_stats(model_dir, device, model_details):
    model_dict = torch.load(model_dir / model_details["best_checkpoint"])
    model = model_dict["model"].to(device)

    val_datapts = build_model_datapoints(Macros.data_dir / model_details["data_name"] / "val")
    test_datapts = build_model_datapoints(Macros.data_dir / model_details["data_name"] / "test")

    val_label_map, val_pred_map = get_binary_preds(model, device, val_datapts)
    test_label_map, test_pred_map  = get_binary_preds(model, device, test_datapts)

    final_map = {}
    metric_map_val, final_threshold = compute_binary_metrics(val_label_map, val_pred_map)
    metric_map_test, _ = compute_binary_metrics(test_label_map, test_pred_map, final_threshold)
    final_map["val"] = metric_map_val
    final_map["test"] = metric_map_test
    return final_map



def eval_all_models(config, device):
    model_dir = Path(config.models_dir)
    aggregate_stats = {}
    for model in MODEL_CONFIG_DICT:
        model_details = MODEL_CONFIG_DICT[model]

        if model_details["type"] == "unary":
            aggregate_stats[model] = compute_unary_stats(model_dir, device, model_details)

        if model_details["type"] == "binary":
            aggregate_stats[model] = compute_binary_stats(model_dir, device, model_details)

        if model_details["type"] == "pmt_baseline":
            aggregate_stats[model] = compute_pmt_stats(model_dir, device, model_details)
    
    with open("results.json", "w") as f:
        json.dump(aggregate_stats, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--models_dir", type=str, default=Macros.model_dir)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(Macros.random_seed)

    # if not os.path.exists(os.path.dirname(model_save_dir)):
    eval_all_models(args, device)
