#!/usr/bin/env python

# 在本地机器上设置defects4j的脚本
# IMPORTANT: to run this script you must have cpanm installed

from pathlib import Path
import os

curr_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
defects4j_dir: Path = curr_dir.parent.parent

def setup_defects4j() -> None:
    cwd: str = os.getcwd()
    os.chdir(defects4j_dir)

    if not (defects4j_dir / "defects4j").is_dir():
        os.system("git clone http://github.com/rjust/defects4j")
        
    os.chdir("defects4j")
    os.system("git checkout v2.0.0")
    os.system("cpanm --installdeps .")
    os.system("./init.sh")

    defects_4j_bin = defects4j_dir / "defects4j" / "framework" / "bin"
    print(f"Add following to path in your .bashrc file: export PATH=$PATH:{defects_4j_bin}")
    os.chdir(cwd)

if __name__ == "__main__":
    setup_defects4j()