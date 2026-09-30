# PulseLab · 心电智能分析

本地运行的心电信号分析工作站，当前应用版本为 2.1.0。采用 Python FastAPI、原生 HTML/CSS/JavaScript、SQLite 和 ONNX Runtime CPU 推理。

## 功能

- CSV/JSON 心电导入与多导联波形显示。
- 滤波、R 峰检测、心率与 RR 间期分析、频谱和心搏模板。
- 选段分析、记录管理、人工复核、记录对比和报告导出。
- 导入 ONNX 模型及 JSON 配置，执行心搏分类、整段分类或多标签推理。
- 提供 MIT-BIH 数据检查、训练数据准备、模型训练与导出脚本。

## 启动

需要 Python 3.10 或更新版本。

Windows 下双击 `启动心电分析.cmd`，或在项目目录运行：

```powershell
python start.py
```

启动程序会创建独立的 `.venv` 环境并安装项目依赖，首次安装需要联网。默认打开 <http://127.0.0.1:8765>，服务只监听本机地址。运行期间保持启动窗口开启，按 `Ctrl+C` 停止。

可指定端口或关闭自动打开浏览器：

```powershell
python start.py --port 8766 --no-browser
```

## 数据与模型

仓库包含公开数据的短片段与明确标注的合成示例；来源、处理方式和许可见 `data/数据来源与许可.txt`。完整 MIT-BIH 原始数据包不纳入 Git，可从 [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) 下载，将数据文件放到 `mit-bih-arrhythmia-database/mit-bih-arrhythmia-database-1.0.0/`。

可重新准备应用演示数据：

```powershell
.venv\Scripts\python.exe scripts/prepare_data.py
```

在“模型管理”页面导入单个、权重内嵌的 ONNX 文件及匹配的 JSON 配置。配置示例见 `模型配置示例.json`；实际配置必须与模型的类别顺序、导联、输入长度及预处理方式一致。

训练依赖与应用依赖分开管理。需要训练时创建独立环境，并查看脚本参数：

```powershell
python -m venv .training-venv
.training-venv\Scripts\python.exe -m pip install -r requirements-training.txt
.training-venv\Scripts\python.exe scripts/inspect_training.py --help
.training-venv\Scripts\python.exe scripts/prepare_mitbih_training.py --help
.training-venv\Scripts\python.exe scripts/train_mitbih.py --help
```

## 开发验证

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

`backend/` 为后端与分析引擎，`web/` 为前端，`scripts/` 为数据与训练工具，`tests/` 为自动化测试。用户导入记录保存在本地 `storage/`，虚拟环境、本地数据库、原始数据缓存和验证报告已通过 `.gitignore` 排除。

本项目用于科研、教学与信号复核。工程指标和模型输出未经临床验证，不用于诊断、治疗或急救决策。MIT-BIH 数据的许可不等同于本项目代码的许可。
