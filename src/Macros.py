import os
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    src_dir: Path = this_dir
    project_dir: Path = src_dir.parent

    data_dir: Path = project_dir / "data"
    results_dir: Path = project_dir / "_results"
    defects4j_root_dir: Path = data_dir
    defects4j_data_dir: Path = data_dir / "raw_data"
    defects4j_pmt_save_dir: Path = data_dir / "pmt_parsed"
    defects4j_trans_save_dir: Path = data_dir / "trans_parsed"

    downloads_dir: Path = project_dir / "_downloads"
    model_dir: Path = project_dir / "models"
    defects4j_model_dir: Path = model_dir
    model_configs_dir: Path = project_dir / "src" / "model_configs"

    latest_versions: dict = {"Lang": 1, "Chart": 1, "Gson": 15, "Cli": 30, "JacksonCore": 25, "Csv": 15}

    default_train_percentage: int = 80
    default_validation_percentage: int = 10
    default_test_percentage: int = 10
    
    max_mutator_size = 20
    max_method_name = 20

    class_weights = [1, 6.14] # 86% of data is 0, 14% is 1 (0.86/0.14 = 6.14)
    class_weights_diff = [1, 2.57] # 72% of data is 0, 28% is 1 (0.72/0.28 = 2.57)
    class_weights_suite = [1.56, 1] # 39% of data is 0, 61% is 1 (0.61/0.39 = 1.56)
    random_seed = 10

    NUM_MUTATORS = 11

    MODEL_DICT = {"codebert": {"tokenizer": AutoTokenizer, "model": AutoModel, "max_embedding_size": 512,"pretrained":"microsoft/codebert-base"}}
