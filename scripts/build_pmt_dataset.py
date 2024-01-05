#!/usr/bin/env python

"""
Assumes defects4j is setup. you can either run setup_defects4j.py or 
follow instructions here https://github.com/rjust/defects4j

Also need to fulfill these requirements:
Java 1.8 (version 1.5.0 and older requires Java 1.7)
Git >= 1.9
SVN >= 1.8
Perl >= 5.0.12
"""

import os
from pathlib import Path

VERSION_MAP: dict = {
    "Lang": [1],
    # "Lang": [1, 10, 20, 30, 40, 50, 60],
    # "Time": [1, 5, 10, 15, 20, 25],
    # "Chart": [1, 5, 10, 15, 20, 25],
    # "Gson": [15, 10, 5, 1],
    # "Cli": [30, 20, 10, 1],
    # "JacksonCore": [25, 20, 15, 10, 5, 1],
    # "Csv": [15, 10, 5, 1]
}

curr_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
dataset_dir: Path = curr_dir.parent / "_downloads" / "pmt_dataset"

def make_dirs() -> None:
    cwd: str = os.getcwd()
    os.chdir(dataset_dir)

    for proj in VERSION_MAP:
        for version in VERSION_MAP[proj]:
            if not (dataset_dir / f"{proj}_{version}_fixed").is_dir():
                os.mkdir(f"{proj}_{version}_fixed")
    
    os.chdir(cwd)

def build_dataset() -> None:
    cwd: str = os.getcwd()

    for proj in VERSION_MAP:
        for version in VERSION_MAP[proj]:
            version_dir = dataset_dir / f"{proj}_{version}_fixed"
            os.system(f"defects4j checkout -p {proj} -v {version}f -w {version_dir}")
    
    os.chdir(cwd)
    os.system("rm -rf defects4j")



if __name__ == "__main__":
    make_dirs()
    build_dataset()