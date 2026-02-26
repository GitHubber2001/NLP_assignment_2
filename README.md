# NLP_assignment_2
This is the repository for the second NLP assignment.

# Dataset
AG News:
- Link: https://huggingface.co/datasets/sh0416/ag_news
- Version: Version 3, Updated 09/09/2015

# Team
Assignment group 3
- Kevin Kuipers (s5051150)
- Federico Berdugo Morales (s5363268)
- Nik Skouf (s5617804)

# Reproducibility notes
The training, evaluation metrics, and error analysis outputs given by the program are deterministic due to the fixed random seed used. To reproduce our results use the Bash commands listed below in sequence in a terminal to download the repository and run the program. To execute these commands successfully you need to have Git, the uv package manager and Python 3.12 or higher installed.

## Windows
```console
git clone https://github.com/GitHubber2001/NLP_assignment_2
cd NLP_assignment_2
uv sync
source .venv/Scripts/activate
python main.py --debug=false
```

## Linux and macOS
```console
git clone https://github.com/GitHubber2001/NLP_assignment_2
cd NLP_assignment_2
uv sync
source .venv/bin/activate
python main.py --debug=false
```
