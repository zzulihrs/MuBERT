import argparse
import os, sys
import random
import numpy as np
from tqdm import tqdm
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from Macros import Macros
from preprocessing import utils
import pickle

# Method that splits list into train validation and test sets
def split_three(lst, args):
    train_r, val_r, test_r = args.train_percentage/100, args.validation_percentage/100, args.test_percentage/100

    indicies_for_splitting = [int(len(lst) * train_r), int(len(lst) * (train_r+val_r))]
    train, val, test = np.split(lst, indicies_for_splitting)
    return train, val, test

# Method that randomly samples and splits list of mutants
def random_sample_mutants(mutants, args):
    random.shuffle(mutants)
    train, val, test = split_three(mutants, args)

    return train, val, test

def split_cp(mutants):
    cp_train = []
    cp_train += mutants["Lang"]
    cp_train += mutants["Chart"]
    cp_train += mutants["JacksonCore"]

    cp_val = []
    cp_val += mutants["Cli"]
    cp_val += mutants["Gson"]

    cp_test = []
    cp_test += mutants["Csv"]

    return cp_train, cp_val, cp_test

# Primary method respnosible for tokenizing method and mutated line
def tokenize_str(method, test, new_line, line_idx, tokenizer):
    tokens = [tokenizer.cls_token]
    # Add method tokens to list with seperator token between lines
    for i in range(len(method)):
        if i == line_idx:
            tokens += ["<BEFORE>"]
            tokens += tokenizer.tokenize(method[i])
            tokens += ["<AFTER>"]
            tokens += tokenizer.tokenize(new_line)
            tokens += ["<ENDDIFF>"]
        else:
            tokens += tokenizer.tokenize(method[i])
    
    tokens += [tokenizer.sep_token]

    for i in range(len(test)):
        tokens += tokenizer.tokenize(test[i])
    tokens += [tokenizer.eos_token]

    ids =  tokenizer.convert_tokens_to_ids(tokens) 
    mask = [1] * (len(tokens))
    
    if len(tokens) > 512:
        return [], [], -1

    padding_length = 512 - len(tokens)
    ids += [tokenizer.pad_token_id]*padding_length
    mask+=[0]*padding_length
    return ids, mask, 0

def subsample_mutants(mutants, test_map, tokenizer, prefix, is_cp):
    new_mutants = []
    i = 0
    num_skipped = 0
    key_set = set()
    for ind, mutant in enumerate(tqdm(mutants)):
        key = str(mutant["mut_src_line_no"])+mutant["before"]+mutant["after"]+mutant["class_name"]+mutant["method_name"]
        if key in key_set:
            continue
        key_set.add(key)

        curr_suite = {"label": mutant["label"], "mutants": []}
        for test in mutant["tests"]:
            if mutant["mut_src_line_no"] >= len(mutant["src_lines"]):
                continue
            
            mutated_src_lines = mutant["src_lines"]
            new_line = None
            
            if "NOOP" in mutant["after"]:
                if mutant["mutator"] != "STD":
                    print(mutant)
                else:
                    new_line = ""
            else:        
                new_line = mutated_src_lines[mutant["mut_src_line_no"]].replace(mutant["before"], mutant["after"], 1)
                no_space_before = mutant["before"].replace(" ", "")
                lowercase_before = mutant["before"].replace("F", "f")
                no_space_after = mutant["after"].replace(" ", "")
                lowercase_after = mutant["after"].replace("F", "f")

                if new_line == mutant["src_lines"][mutant["mut_src_line_no"]]:
                    new_line = mutated_src_lines[mutant["mut_src_line_no"]].replace(no_space_before, no_space_after, 1)

                if new_line == mutant["src_lines"][mutant["mut_src_line_no"]]:
                    new_line = mutated_src_lines[mutant["mut_src_line_no"]].replace(lowercase_before, lowercase_after, 1)

                if new_line == mutant["src_lines"][mutant["mut_src_line_no"]]:
                    new_line = None

            if new_line is None:
                continue

            embed, mask, index = tokenize_str(mutated_src_lines, test_map[test], new_line, mutant["mut_src_line_no"], tokenizer)
            if index == -1:
                continue

            if embed[index] != 0:
                num_skipped += 1
                continue

            new_mutant = {
                "mut_no": mutant["mut_no"],
                "test_method": test,
                "source_method": mutant["method_name"],
                "src_lines": mutant["src_lines"],
                "new_line": new_line,
                "tst_lines": test_map[test],
                "before_pmt": mutant["before_pmt"],
                "after_pmt": mutant["after_pmt"],
                "mutator": mutant["mutator"],
                "mut_src_line_no": mutant["mut_src_line_no"],
                "label": 1 if test in mutant["killing_tests"] else 0
            }
            curr_suite["mutants"].append(new_mutant)
        
        if len(curr_suite["mutants"]) > 0:
            random.shuffle(curr_suite["mutants"])
            new_mutants.append(curr_suite)

        if len(new_mutants) == 200:
            random.shuffle(new_mutants)
            set_name = "base_set_suite_cp" if is_cp else "base_set_suite"
            with open(Macros.defects4j_root_dir / set_name / f"{prefix}" / f"{prefix}_{str(i)}", "wb") as f:
                pickle.dump(new_mutants, f) 
            new_mutants = []
            i += 1

    random.shuffle(new_mutants)
    print(num_skipped)
    print(i * 10_000 + len(new_mutants))
    set_name = "base_set_suite_cp" if is_cp else "base_set_suite"
    with open(Macros.defects4j_root_dir / set_name / f"{prefix}" / f"{prefix}_{str(i)}", "wb") as f:
        pickle.dump(new_mutants, f) 

def main(args, tokenizer):
    with open(args.mutants_file, "rb") as f:
        mutants = pickle.load(f) 

    with open(args.test_file, "rb") as f:
        test_map = pickle.load(f) 
    
    set_name = "base_set_suite_cp" if args.is_cp else "base_set_suite"
    (Macros.defects4j_root_dir / set_name / "train").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / set_name / "val").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / set_name / "test").mkdir(exist_ok=True, parents=True)

    mu_train, mu_val, mu_test = split_cp(mutants) if args.is_cp else random_sample_mutants(mutants, args)
    subsample_mutants(mu_train, test_map, tokenizer, "train", args.is_cp)
    subsample_mutants(mu_val, test_map, tokenizer, "val", args.is_cp)
    subsample_mutants(mu_test, test_map, tokenizer, "test", args.is_cp)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutants_file", help="where mutants are located", default=Macros.defects4j_root_dir / "mutants.pkl")
    parser.add_argument("--test_file", help="where test mapping is located", default=Macros.defects4j_root_dir / "test_map.pkl")

    parser.add_argument("--train_percentage", help="train data split", type=int, default=Macros.default_train_percentage)
    parser.add_argument("--validation_percentage", help="validation data split", type=int, default=Macros.default_validation_percentage)
    parser.add_argument("--test_percentage", help="test data split", type=int, default=Macros.default_test_percentage)
    parser.add_argument("--model", type=str, choices=["codebert"], default="codebert")
    parser.add_argument("--is_cp", help="whether cross project or not", action="store_true")

    args = parser.parse_args()

    tokenizer_dict = Macros.MODEL_DICT[args.model]
    tokenizer = tokenizer_dict["tokenizer"].from_pretrained(tokenizer_dict["pretrained"])

    utils.set_seed(Macros.random_seed)
    main(args, tokenizer)
