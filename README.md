# 上下文预测突变测试



该存储库包含用于“上下文预测变异测试”的模型、数据集和预处理脚本。



## 安装环境



要安装所有的python依赖项，可以参考根目录中的conda环境：environment.yml。



## 数据集



**再现步骤：**



用于下载数据集的脚本都在脚本目录中可用。



-setup_defects4j.py-在本地计算机上设置defects4j的脚本



**运行步骤**



要构建变体，可以运行：python3 build_mutats.py（如果是跨项目python3build_mutants.py--is_cp）



该脚本将输出两个文件，一个所有变体的列表，以及从测试名称到测试代码的映射。



可以运行的测试矩阵：



测试矩阵：python3 build_mutant_set.py（如果跨项目python3build_mmutants_set.py--is_cp--mutants_file<mutants文件的路径>--Test_file<PATH TO Test MAP>）



**运行步骤**



存储库与所有预处理的输出捆绑在一起。



## 模型配置



模型配置在src/Model_configs下



**复制步骤**

要训练模型，请在（运行时/传输）下运行以下命令：



python train.py--模型配置<模型配置路径>--实验<实验名称>--训练路径<训练路径>--验证路径<验证路径>--is_diff





为了选择套件检查点，我们计算了在测试套件验证数据集上评估时验证F1得分最好的矩阵检查点。



## 评估



要评估模型，需要运行以下两个文件之一：



-eval_models.py-矩阵求值脚本

-eval_suite.py-用于评估套件的脚本