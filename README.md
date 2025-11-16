# TCLaaC - The Command Line as a Corpus

A Natural Language Processing (NLP) approach to analyzing Windows Sysmon Event ID 1 (Process Creation) logs using Latent Dirichlet Allocation (LDA) topic modeling.

## Overview

This project applies topic modeling to command-line data extracted from Sysmon logs to:
- **Discover patterns** in command-line usage across large datasets
- **Identify anomalous behavior** by detecting rare command patterns
- **Group similar commands** into interpretable topics
- **Detect potentially malicious activity** using LOLBAS (Living Off The Land Binaries and Scripts) enrichment

## Features

- **Advanced Text Preprocessing**: Normalization rules for GUIDs, IPs, hex strings, dates, and more
- **Optimized Tokenization**: Custom regex-based tokenizer for command-line syntax
- **Parallel Processing**: Multi-core support for fast processing of millions of logs
- **Hyperparameter Tuning**: Automated LDA optimization using coherence scores
- **Interactive Visualization**: Treemaps and topic explorers for result analysis
- **LOLBAS Integration**: Enrichment with known dual-use binaries for security analysis

## Project Structure

```
TCLaaC/
├── main.py                     # Streamlined pipeline (new)
├── config.py                   # Centralized configuration (new)
├── data_loader.py             # CSV/synthetic data loading (new)
├── helpers.py                 # Core preprocessing functions
├── graphs.py                  # Visualization utilities
├── requirements.txt           # Python dependencies
├── The Command Line as a Corpus.ipynb  # Original research notebook
├── OSBinaries/                # LOLBAS YAML files
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/CampbellTrevor/TCLaaC.git
cd TCLaaC
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Using the Streamlined Pipeline (main.py)

```bash
python main.py --input data.csv --output results/ --topics 11
```

### Option 2: Using the Jupyter Notebook

```bash
jupyter notebook "The Command Line as a Corpus.ipynb"
```

## Data Input Formats

### CSV Format
Your CSV file should have a column named `command_line` containing the command-line strings:

```csv
command_line
"powershell.exe -ExecutionPolicy Bypass -File script.ps1"
"cmd.exe /c whoami"
"C:\Windows\System32\net.exe user admin P@ssw0rd /add"
```

### Synthetic Data Generation
For testing without real data:

```python
from data_loader import generate_synthetic_data
data = generate_synthetic_data(num_samples=10000)
```

## Configuration

Edit `config.py` to customize:
- **Model Parameters**: Number of topics, random state, LDA hyperparameters
- **Preprocessing Rules**: Normalization patterns, tokenization settings
- **Performance Settings**: Number of CPU cores, batch sizes
- **File Paths**: Input/output locations, LOLBAS repository path

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_TOPICS` | 11 | Number of latent topics to discover |
| `RANDOM_STATE` | 42 | Seed for reproducibility |
| `LOLBAS_REPO_PATH` | './OSBinaries' | Path to LOLBAS YAML files |
| `MIN_DOC_LENGTH` | 2 | Minimum tokens per document |
| `COHERENCE_METHOD` | 'c_v' | Coherence calculation method |

## Workflow

1. **Data Loading**: Load Sysmon logs from CSV, database, or generate synthetic data
2. **Preprocessing**: 
   - Extract command lines from Sysmon messages
   - Normalize using regex rules (replace GUIDs, IPs, etc.)
   - Tokenize into meaningful units
   - Identify root executables
3. **LOLBAS Enrichment**: Add known malicious command examples
4. **Feature Engineering**: Convert to TF-IDF vectors
5. **Model Training**: Train LDA with optimized hyperparameters
6. **Topic Assignment**: Assign each command to its most likely topic
7. **Visualization**: Generate interactive treemaps and topic explorers
8. **Analysis**: Identify suspicious topics using LOLBAS keyword scoring

## Performance Tips

- **Large Datasets (>1M rows)**: Use `main.py` instead of the notebook for better memory management
- **Hyperparameter Tuning**: Can be time-consuming; use pre-tuned values for production
- **Parallel Processing**: Automatically uses all available CPU cores
- **Memory Usage**: ~2-4GB RAM per million command lines

## Output Files

- `lda_model.joblib`: Trained LDA model (reusable)
- `analysis_dataframe.parquet`: Full results with topic assignments
- `topic_treemap.html`: Interactive visualization (if enabled)
- `topic_summary.csv`: Topic keywords and statistics

## Advanced Usage

### Custom Normalization Rules

Add rules in `config.py`:

```python
NORMALIZATION_RULES_COMPILED = {
    'custom_pattern': (
        re.compile(r'your-regex-here', re.IGNORECASE),
        '<YOUR_PLACEHOLDER>'
    ),
}
```

### Hyperparameter Tuning

```python
from main import tune_hyperparameters
best_params = tune_hyperparameters(
    corpus=corpus,
    dictionary=dictionary,
    num_topics_range=(5, 50, 2)
)
```

## LOLBAS Data

The `OSBinaries/` folder contains YAML files from the [LOLBAS Project](https://lolbas-project.github.io/). These files document Windows binaries that can be abused for malicious purposes while appearing legitimate.

To update LOLBAS data:
```bash
# Manual: Download latest from https://github.com/LOLBAS-Project/LOLBAS
# Or use git submodule (if configured)
```

## Troubleshooting

### Common Issues

**Import Error: `ionic_scripting_framework`**
- This is a custom module for database querying
- Solution: Use CSV input or synthetic data (see `data_loader.py`)

**Memory Error during training**
- Reduce dataset size or use sampling
- Decrease `chunksize` in LDA parameters

**Low Coherence Scores**
- Try different numbers of topics
- Increase `passes` in LDA training
- Check data quality (too many short/duplicated commands)

## Research & Citation

This project demonstrates the application of NLP topic modeling to security log analysis, specifically:
- Command-line behavioral clustering
- Anomaly detection through statistical modeling
- Integration of threat intelligence (LOLBAS) with unsupervised learning

If you use this work in your research, please cite:
```
@software{tclaaC2025,
  author = {Campbell, Trevor},
  title = {TCLaaC: The Command Line as a Corpus},
  year = {2025},
  url = {https://github.com/CampbellTrevor/TCLaaC}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## Acknowledgments

- [LOLBAS Project](https://lolbas-project.github.io/) for dual-use binary documentation
- Gensim library for LDA implementation
- The information security community for inspiration

## Contact

- GitHub: [@CampbellTrevor](https://github.com/CampbellTrevor)
- Issues: [GitHub Issues](https://github.com/CampbellTrevor/TCLaaC/issues)

---

**Last Updated**: November 2025
