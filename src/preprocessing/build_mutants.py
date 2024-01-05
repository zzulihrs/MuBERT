import argparse
import json
import os, sys
import random
import numpy as np

from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import re
import javalang
import pickle
import codecs

from xml.dom import minidom
from glob import glob
from Macros import Macros


# 目的：将文件路径转换为点分隔的路径（通常用于Java包命名）。
# 用法：传入文件路径和需要移除的子路径，将其转换为点分隔路径。
# Method to convert path into dot path
def convert_to_dot(path, subpath):
    return path.replace(subpath, "").replace(".java", "").replace("/", ".")

# 目的：更新方法映射表，用于记录Java文件中方法的开始和结束行号。
# 用法：传入点分隔的路径、方法标识、方法映射表和开始、结束行号来更新映射。
def update_method_map(dot_path, is_method, method_map, start, end):
    if is_method:
        method_map.setdefault(dot_path, []).append((start, end))
    else:
        method_map[dot_path] = (start, end)

# 目的：构建一个方法映射表，用于从Java源代码树中提取方法信息。
# 用法：给定Java解析树、文件路径、基路径和方法标识，提取方法信息并建立映射。
# Method that takes java tree object and returns a list of relevant lines
# Each line is a tuple of (node_name, method_name, start, end, params, str_params)
def build_method_mapping(tree, file_path, base_path, is_method):
    method_map = {}
    for _, node in tree.filter(javalang.tree.TypeDeclaration):
        # get method lines
        for method in node.methods:
            if hasattr(method, "body") and method.body is not None and len(method.body) > 0:
                method_start_line = method.position.line
                # Calculate method end line by finding the max line number in the method body
                method_end_line = max(stmt.position.line for stmt in method.body if hasattr(stmt, 'position'))

                dot_path = convert_to_dot(file_path, base_path + "/") if is_method else convert_to_dot(file_path, base_path + "/") + "." + method.name
                update_method_map(dot_path, is_method, method_map, method_start_line, method_end_line)
        # get constructor lines
        for const in node.constructors:
            if hasattr(const, "body") and const.body is not None and len(const.body) > 0:
                const_start_line = const.position.line
                const_end_line = max(stmt.position.line for stmt in const.body if hasattr(stmt, 'position'))
                dot_path = convert_to_dot(file_path, base_path + "/") if is_method else convert_to_dot(file_path, base_path + "/") + "." + node.name
                update_method_map(dot_path, is_method, method_map, const_start_line, const_end_line)
    return method_map


# 目的：对源代码字符串进行标记化，仅考虑之前的上下文。
# 用法：给定上下文行、当前行、变异器、最大长度和标记器，生成标记化的源代码。
# Method for tokenizing string that only takes preceding context into account
def tokenize_source_str_previous(context_lines, line, mutator, max_length, tokenizer):
    context = [item for sublist in context_lines for item in sublist]
    source_tokens = context + [tokenizer.sep_token] + line + [tokenizer.sep_token] + tokenizer.tokenize(mutator)+ [tokenizer.sep_token]
    source_tokens = source_tokens[-(max_length-1):]
    source_tokens = [tokenizer.cls_token] + source_tokens
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_mask = [1] * (len(source_tokens))
    padding_length = max_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    source_mask += [0]*padding_length
    return source_ids, source_mask

# Primary method respnosible for tokenizing method and mutated line
def tokenize_str(method, max_length, tokenizer, add_cls_tokens, line_no=None):
    source_tokens = [tokenizer.cls_token]
    index = -1

    # Add method tokens to list with seperator token between lines
    for i in range(len(method)):
        source_tokens += method[i] + [tokenizer.sep_token]
        if add_cls_tokens:
            source_tokens += [tokenizer.cls_token]
        if line_no is not None and i == line_no-1:
            index = len(source_tokens) - 1
    
    if len(source_tokens) > max_length:
        # If line number is specified create sliding window of k tokens around the line
        if line_no is not None:
            new_tokens = [source_tokens[index]]
            target_length = max_length - 2 # -2 for cls, sep
            curr_window = 1
            new_index = 0
            while len(new_tokens) < target_length and ((index - curr_window > 0) or (index + curr_window) < len(source_tokens)):
                if index - curr_window >= 0:
                    new_tokens= [source_tokens[index-curr_window]] + new_tokens 
                    new_index += 1
                if len(new_tokens) < target_length and index + curr_window < len(source_tokens):
                    new_tokens= new_tokens + [source_tokens[index+curr_window]]
                curr_window += 1
            index = new_index
            source_tokens = [tokenizer.cls_token] + new_tokens + [tokenizer.sep_token]
        # Otherwise just take the first 512 tokens of the method
        else:
            source_tokens = source_tokens[:max_length-1] + [tokenizer.sep_token]
    
    # Convert tokens to ids and padding
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
    source_mask = [1] * (len(source_tokens))
    padding_length = max_length - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    source_mask+=[0]*padding_length
    return source_ids, source_mask, index

# 在给定的 Java 文件中搜索方法和构造函数，返回一个映射，其中包含文件中每个方法或构造函数的起止行号。
# Method that searches file path for list of methods and returns the list of lines for that file that map to methods
def search_method(file_path, base_path, is_method):
    with open(file_path, "r") as f:
        try:
            tree = javalang.parse.parse(f.read())
        except:
            print("Error: " + file_path)
            return None

    return build_method_mapping(tree, file_path, base_path, is_method)

# 从 pickle 文件中读取覆盖映射并转换为更易读的格式。通常用于分析测试覆盖率数据。
# Method to serialize and convert coverage map to more readable format
def pickle_cov_map_to_cov_map(pickle_path):
    with open(pickle_path, "rb") as f:
        cov_map_df = pickle.load(f)
    
    cov_map_df['sum'] = cov_map_df[list(cov_map_df)].sum(axis=1)

    cov_map = {}
    for index, row in cov_map_df.iterrows():
        file_name = index[0].split("$")[0]
        if file_name not in cov_map:
            cov_map[file_name] = {}
        
        cov_map[file_name][int(index[1])] = row['sum']
    
    return cov_map

# 处理测试数组，返回一个包含测试类名称的列表。
# Method that porcesses list of tests to get a list of test classes
def process_tests_arr(tests_arr):
    return [elem.split("(")[0].split(")")[0] for elem in tests_arr]

# 这些函数与测试数据的处理、突变体测试的建立和数据结构的创建有关。
# Method that takes test mapping and tokenizer and adds test to map of tests and their tokens
def build_mutant_tests_tokens(tests, test_line_map, tot_test_map, proj_name, proj_no):
    new_tests = []
    for test in tests:
        # if the test does not exist in the mapping, try to find another test with same name in the package
        if test not in test_line_map:
            # Try to find method with same name in package
            package = test.rsplit(".", 2)[0]
            found_test = False
            for key in test_line_map:
                if package in key and test.rsplit(".", 1)[1] in key:
                    test = key
                    found_test = True
                    break
            if not found_test:
                continue
        
        # If in the map, append the test to the list of tests that should be saved
        if test in tot_test_map:
            new_tests.append(test)
            continue

        # Otherwise we need to parse the test and add it to the map
        curr_test_arr = []
        start_line, end_line = test_line_map[test]
        file_name = test.rsplit(".", 1)[0]
        json_file_path = Macros.defects4j_data_dir / "json_files_trans/{}_json/{}_{}_fixed/{}/{}".format(proj_name, proj_name, proj_no, "test", f"{file_name}.java")
        
        if not os.path.exists(json_file_path):
            print("Not found: ", test)
            continue
                
        with open(json_file_path, encoding='utf-8', errors='ignore', mode="r") as f:
            lines = json.load(f)
        
        for k in range(start_line-1, end_line):
            curr_test_arr.append(lines[k])
        
        if test not in tot_test_map and  len(curr_test_arr) > 0:
            tot_test_map[test] = curr_test_arr
    return new_tests

def build_data_order(mutants):
    data_order = []
    for idx, mutant in enumerate(mutants):
        for test in mutant["tests"]:
            status = 1 if test in mutant["killing_tests"] else 0
            data_order.append({"mutant_idx": idx, "test": test, "status": status})
    random.shuffle(data_order)
    return data_order

def build_total_map(base_path, files, is_method):
    map_tot = {}
    for path in files:
        if path[-4:] != "java":
            continue
        map = search_method(path, base_path, is_method)
        if map is not None:
            map_tot = dict(map_tot, **map)
    return map_tot

# 解析记录突变体测试结果的日志文件，建立突变体和相关测试的详细信息映射。
# Major 是一个用于进行突变测试（Mutation Testing）的工具。在软件测试领域，突变测试是一种用于评估软件测试质量的技术，它通过引入小的、有意的错误（称为“突变”）到程序的源代码中，然后检查测试套件是否能够检测出这些引入的错误。
def parse_major_log(major_log_path, proj_name, proj_no, method_map, test_map_major, tot_test_map_major):

    def to_test_name(l):  # 将测试编号列表转换为测试名列表
        new_l = []
        for test_no in l:
            test_name = test_map[test_no].replace('[', '.').replace(']', '')
            new_l.append(test_name)
        return new_l


    kill_reason = ['FAIL', 'TIME', 'EXC']  # kill_reason: 定义了造成突变体死亡的原因列表。
    #  读取和解析 testMap.csv, covMap.csv, killMap.csv，建立测试映射、覆盖映射和杀死映射。
    test_map = {}
    with open(os.path.join(major_log_path, 'testMap.csv')) as tm_file:
        tm_file.readline()
        for line in tm_file.readlines():
            test_no, test_name = tuple(line.strip().split(','))
            test_map[int(test_no)] = test_name
    cov_map = {}
    with open(os.path.join(major_log_path, 'covMap.csv')) as f:
        f.readline()
        for line in f.readlines():
            test_no, mut_no = tuple(line.strip().split(','))
            if int(mut_no) not in cov_map:
                cov_map[int(mut_no)] = []
            cov_map[int(mut_no)].append(int(test_no))

    kill_map = {}
    with open(os.path.join(major_log_path, 'killMap.csv')) as km_file:
        km_file.readline()
        for line in km_file.readlines():
            test_no, mut_no, reason = tuple(line.strip().split(',')[:3])
            test_no = int(test_no)
            mut_no = int(mut_no)

            if reason in kill_reason:
                if mut_no not in kill_map:
                    kill_map[mut_no] = []
                kill_map[mut_no].append(test_no)


    mutants = {}
    with open(os.path.join(major_log_path, 'mutants.log'), newline='') as m_file:
        total_muts = 0
        i = 0
        json_file_name = None
        
        test_ct, source_ct = 0, 0 
        for line in m_file:
            total_muts += 1
            mutant_info = line.strip().split(':')
            mut_no = int(mutant_info[0])
            class_name = mutant_info[4]
            try:
                line_no = int(mutant_info[5])
            except Exception as e:
                continue

            split_class_name = class_name.split("@")[0].split("$")[0]
            split_class_name = 'D:\\download\\xunlei\\contextual-pmt-artifact\\_downloads\\pmt_dataset\\Lang_1_fixed.src.main.java' + "\\" + split_class_name.replace('.', '\\')
            # If we can't find class name, skip building this mutant
            if split_class_name not in method_map:
                continue
        
            # Search for method in list of method lines in class
            start_line, end_line = -1, -1
            for start, end in method_map[split_class_name]:
                if line_no-1 >= start and line_no-1 <= end:
                    start_line, end_line = start, end
                    break

            src_method_key = f"{split_class_name}_{start_line}_{end_line}"

            if mut_no not in cov_map:
                continue
            if "@" not in class_name:
                continue

            method_name = class_name.split("@")[1]
            class_name = class_name.split("@")[0]
            if "$" in class_name:
                di = class_name.index("$")
                json_file_name = class_name[:di] + ".java"
            else:
                json_file_name = class_name + ".java"

            json_file_path = Macros.defects4j_data_dir / "json_files_trans/{}_json/{}_{}_fixed/{}/{}".format(proj_name, proj_name, proj_no, "src", json_file_name)

            if not os.path.exists(json_file_path):
                print("Not found: ", json_file_name)
                continue

            changed = ":".join(mutant_info[6:])
            mutants[mut_no] = {
                'class_name': class_name,
                "mut_no": mut_no,
                'method_name': method_name,
                'line_no': line_no,
                "src_method_key": src_method_key,
                "status": "UNKNOWN",
                "label": -1,
                "tests": [],
                "killing_tests": [],
                "passing_tests": [],
                'mutator': mutant_info[1],
                "src_lines": [],
                "mut_src_line_no": -1,
                'before': "",
                'after': "",
                'before_pmt': "",
                'after_pmt': "",
                'body': "",
                'matching_idx': None
            }
            # if "$" in class_name:
            #    print(mutants[mut_no]["body"])
            lines = []
            with open(json_file_path, encoding='utf-8', errors='ignore', mode="r") as f:
                lines = json.load(f)
            
            curr_method_arr = []
            if end_line+1 > len(lines):
                source_ct += 1
                continue

            for k in range(start_line-1, end_line):
                curr_method_arr.append(lines[k])
            
            if start_line == -1 or len(curr_method_arr) == 0:
                source_ct += 1
                continue

            mutants[mut_no]["src_lines"] = curr_method_arr
            
            mutants[mut_no]["mut_src_line_no"] = line_no-start_line
            mutants[mut_no]["body"] = lines[mutants[mut_no]["line_no"]-1]
            if mut_no not in kill_map:
                mutants[mut_no]["killing_tests"] = []
                mutants[mut_no]["passing_tests"] = to_test_name(cov_map[mut_no])
                mutants[mut_no]["status"] = "SURVIVED"
                mutants[mut_no]["label"] = 0
            else:
                mutants[mut_no]["killing_tests"] = to_test_name(kill_map[mut_no])
                mutants[mut_no]["passing_tests"] = to_test_name(list(set(cov_map[mut_no]) - set(kill_map[mut_no])))
                mutants[mut_no]["status"] = "KILLED"
                mutants[mut_no]["label"] = 1
            
            mutants[mut_no]["tests"] = mutants[mut_no]["killing_tests"] + mutants[mut_no]["passing_tests"]
            mutants[mut_no]["tests"] = build_mutant_tests_tokens(mutants[mut_no]["tests"], test_map_major, tot_test_map_major, proj_name, proj_no)
            if len(mutants[mut_no]["tests"]) == 0:
                test_ct += 1
                continue
            i += 1


    mutant_lexed_file_path = Macros.defects4j_data_dir / "major_mutations" / "{}_{}_fixed/mutants.log.slp".format(proj_name, proj_no)
    with open(mutant_lexed_file_path, "rb") as f:
        for line in f:
            if line[0] != 91:
                continue

            tok = line.strip()[5:-5].decode("utf-8", "ignore").encode("utf-8").decode("utf-8")
            _mut_no = int(tok.split(",")[0][8:])
            content = ",".join(tok.split(",")[1:]).strip().replace("NOOP", "<noop>")

            if _mut_no not in mutants:
                continue

            if tok[1:8] == "JINHANA":
                mutants[_mut_no]["before_pmt"] = content
            else:
                mutants[_mut_no]["after_pmt"] = content

    mutant_lexed_file_path = Macros.defects4j_data_dir / "major_mutations" / "{}_{}_fixed/mutants.log.parsed".format(proj_name, proj_no)
    with open(mutant_lexed_file_path, "r") as f:
        for line in f:
            line_content = line.split("JINHAN")[1].strip()
            tag = line_content[0]
            remainder = line_content[1:]
            _mut_no, content = remainder.split(" ", 1)
            _mut_no = int(_mut_no)

            if _mut_no not in mutants:
                continue
            if tag == "A":
                mutants[_mut_no]["before"] = content.strip()
            else:
                mutants[_mut_no]["after"] = content.strip()

    # 打印总突变体数量和成功解析的数量。
    # 打印无法关联测试和源代码的突变体数量
    print("# of mutants: {} -> {}".format(total_muts, i))
    print(f"test ct: {test_ct}, source ct: {source_ct}")

    mutants = list(mutants.values())
    return mutants


# 这是脚本的主要入口点，它解析命令行参数并处理多个项目。它使用前面定义的函数来解析 Java 代码，生成突变体，处理测试用例等。
def parse_all_projects(args):
    tokenizer_dict = Macros.MODEL_DICT[args.model]  # 从 args.model 指定的模型中，获取 tokenizer 相关的配置和预训练模型。
    tokenizer = tokenizer_dict["tokenizer"].from_pretrained(tokenizer_dict["pretrained"])
    max_size = Macros.MODEL_DICT[args.model]["max_embedding_size"]

    mutants_list = {} if args.is_cp else []  # 突变体列表和测试映射 is_cp：是否跨项目
    tot_test_map = {}

    for proj_name, proj_no in Macros.latest_versions.items():  # 中包含了项目名和版本号
        proj_path = str(Macros.downloads_dir / "pmt_dataset" / f"{proj_name}_{proj_no}_fixed")
        src_class_dir, test_class_dir = "", ""

        with open(str(Macros.defects4j_data_dir / f"src_test_dirs/{proj_name}.txt"), "r") as f:
            for line in f:
                spl = line.strip().split(",")
                if proj_name == spl[0] and str(proj_no) == spl[1]:
                    src_class_dir = spl[2]
                    test_class_dir = spl[3]
                    break

        # 通过读取文件并分割每行来确定源码和测试目录的相对路径
        src_files = [f for f in glob(proj_path + "/" + src_class_dir + "/**", recursive=True) if os.path.isfile(f) and ".java" in f]
        test_files = [f for f in glob(str(proj_path) + "/" + test_class_dir + "/**", recursive=True) if os.path.isfile(f) and "Test" in f]
        method_map = build_total_map(os.path.join(proj_path+"/" ,src_class_dir), src_files, is_method=True)
        test_map = build_total_map(os.path.join(proj_path+"/" ,test_class_dir), test_files, is_method=False)                
        print("Done parse src")
        mutants = parse_major_log(str(Macros.defects4j_data_dir / "major_mutations" / f"{proj_name}_{proj_no}_fixed"), proj_name, proj_no, method_map, test_map, tot_test_map)
        print("Done parse major log")
        if args.is_cp:
            mutants_list[proj_name] = mutants
        else:
            mutants_list += mutants

    save_dir = Macros.defects4j_root_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    mutants_name = "mutants_cp.pkl" if args.is_cp else "mutants.pkl"
    with open(save_dir / mutants_name, "wb") as f:
        pickle.dump(mutants_list, f)

    test_name = "test_map_cp.pkl" if args.is_cp else "test_map.pkl"
    with open(save_dir / test_name, "wb") as f:
        pickle.dump(tot_test_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_type", help="how to sample data", choices=["random"], default="random")
    parser.add_argument("--model", type=str, choices=["codebert"], default="codebert")
    parser.add_argument("--is_cp", help="whether cross project or not", action="store_true")

    random.seed(Macros.random_seed)
    np.random.seed(Macros.random_seed)
    
    args = parser.parse_args()    
    parse_all_projects(args)
