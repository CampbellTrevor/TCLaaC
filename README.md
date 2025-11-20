# TCLaaC - The Command Line as a Corpus

A Natural Language Processing (NLP) approach to analyzing Windows Sysmon Event ID 1 (Process Creation) logs using Latent Dirichlet Allocation (LDA) topic modeling.

## Overview

This project applies topic modeling to command-line data extracted from Sysmon logs to:
- **Discover patterns** in command-line usage across large datasets
- **Identify anomalous behavior** by detecting rare command patterns
- **Group similar commands** into interpretable topics
- **Detect potentially malicious activity** using LOLBAS (Living Off The Land Binaries and Scripts) enrichment

## Features

### NLP & Topic Modeling
- **Enhanced Text Preprocessing**: Comprehensive normalization for URLs, emails, registry keys, UNC paths, domains, GUIDs, IPs, hex strings, dates, timestamps, and file paths
- **Optimized Tokenization**: Custom regex-based tokenizer for command-line syntax with edge case handling
- **Parallel Processing**: Multi-core support for fast processing of millions of logs
- **Hyperparameter Tuning**: Automated LDA optimization using coherence scores
- **Command Complexity Analysis**: Multi-factor scoring for obfuscation detection
- **Data Quality Validation**: Automatic checks for data integrity, duplicates, and statistical anomalies

### Security Analysis
- **LOLBAS Integration**: Enrichment with known dual-use binaries and density-based risk scoring
- **MITRE ATT&CK Mapping**: Automatic technique identification across 9 attack categories
- **Comprehensive Risk Scoring**: Weighted formula combining LOLBAS density, MITRE coverage, complexity, and binary diversity
- **Advanced Anomaly Detection**: Multi-method ensemble approach including:
  - Statistical outlier detection (Z-score based)
  - Complexity-based anomaly scoring
  - Isolation Forest ML algorithm
  - Command sequence pattern analysis
  - Baseline deviation detection
  - Ensemble scoring combining all methods
- **Behavioral Pattern Detection**: Multi-dimensional security analysis per topic

### Visualization & Reporting
- **Comprehensive Index Dashboard**: Main entry point showcasing all analysis results with summary statistics
- **Interactive Analysis Dashboard**: Multi-tab SPA with 8+ visualization types
- **Topic Treemaps**: Hierarchical command grouping with fuzzy matching
- **Security Risk Charts**: LOLBAS density and risk score visualization
- **Word Heatmaps**: Topic-word distribution analysis
- **Distribution Sunbursts**: Proportional topic representation
- **Complexity Box Plots**: Command length distribution by topic
- **Network Graphs**: 
  - Command co-occurrence network showing relationship patterns
  - Topic relationship network with similarity connections
  - MITRE ATT&CK technique co-occurrence network

### Performance & Caching
- **Intelligent Caching**: Automatic caching of expensive operations (LDA models, LOLBAS data, corpus)
- **Cache Management**: Configurable expiration, statistics, and cleanup utilities
- **30-50% Speed Improvement**: On repeated analyses with similar datasets

## Project Structure

```
TCLaaC/
├── main.py                     # Streamlined pipeline with integrated features
├── config.py                   # Centralized configuration
├── data_loader.py             # CSV/synthetic data loading
├── helpers.py                 # Core preprocessing functions
├── graphs.py                  # Visualization utilities
├── cache_manager.py           # Intelligent caching system (new)
├── quality_checker.py         # Data quality validation (new)
├── anomaly_detector.py        # Multi-method anomaly detection (new)
├── network_viz.py             # Network graph visualizations (new)
├── requirements.txt           # Python dependencies
├── test_pipeline.py           # Basic integration tests
├── test_enhanced_features.py  # Feature-specific tests
├── test_new_features.py       # Tests for new capabilities (new)
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

### Option 1: Using the Streamlined Pipeline (main.py) - Recommended

Generate a comprehensive analysis with all features:

```bash
# Analyze with synthetic data (for testing)
python main.py --synthetic 1000 --output results/

# Analyze your own CSV data
python main.py --input your_data.csv --output results/ --topics 11

# Quick analysis without tuning
python main.py --input data.csv --no-tune --output results/
```

After running, open `results/index.html` in your browser to view the comprehensive analysis dashboard.

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

The pipeline generates a comprehensive set of outputs:

### Main Deliverables
- **`index.html`**: 🌟 **Main Dashboard** - Comprehensive entry point with:
  - Summary statistics cards
  - High-risk topic analysis
  - Links to all visualizations (including network graphs)
  - Methodology documentation
  - Key insights
- **`analysis_dashboard.html`**: Interactive SPA with 5 visualization tabs:
  - Topic Treemap (with LOLBAS filtering)
  - Word Heatmap
  - Security Risk Chart
  - Distribution Sunburst
  - Complexity Analysis

### Network Visualizations
- **`command_network.html`**: Command co-occurrence network showing relationship patterns
- **`topic_network.html`**: Topic similarity network with connections based on shared vocabulary
- **`mitre_network.html`**: MITRE ATT&CK technique co-occurrence network

### Data Files
- **`lda_model.joblib`**: Trained LDA model (reusable)
- **`analysis_dataframe.parquet`**: Full results with topic assignments, MITRE techniques, complexity scores, anomaly scores
- **`topic_summary.csv`**: Smart topic names, keywords, and document counts

### Cache Directory (Optional)
- **`.tclaaс_cache/`**: Cached models, LOLBAS data, and intermediate results for faster repeated analyses

### How to Use
1. Run the pipeline: `python main.py --synthetic 1000 --output results/`
2. Open `results/index.html` in your browser
3. Explore interactive visualizations
4. Download data files for further analysis

## Advanced Usage

### Security Analysis Features

#### MITRE ATT&CK Technique Mapping

Commands are automatically mapped to MITRE ATT&CK techniques:
- **T1059**: Command and Scripting Interpreter (PowerShell, CMD, etc.)
- **T1053**: Scheduled Task/Job (schtasks, at.exe)
- **T1105**: Ingress Tool Transfer (certutil, bitsadmin downloads)
- **T1218**: System Binary Proxy Execution (rundll32, regsvr32)
- **T1547**: Boot or Logon Autostart (registry Run keys)
- **T1003**: Credential Dumping (lsass, mimikatz)
- **T1055**: Process Injection
- **T1027**: Obfuscated Files or Information (base64, encoding)

#### Comprehensive Risk Scoring

Each topic receives a risk score calculated as:
```
Risk Score = (LOLBAS Density × 0.4) + 
             (MITRE Coverage × 10 × 0.3) + 
             (Avg Complexity × 0.15) + 
             (Unique Binaries × 2 × 0.15)
```

Where:
- **LOLBAS Density**: Weighted metric of LOLBAS binary usage
- **MITRE Coverage**: Number of unique ATT&CK techniques
- **Avg Complexity**: Command obfuscation/complexity score
- **Unique Binaries**: Diversity of executables in the topic

#### Command Complexity Scoring

Commands are scored (0-100) based on:
- Length and structure
- Special character usage
- Obfuscation indicators (base64, encode, hidden, bypass)
- Command chaining (pipes, redirects, &&)

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
from main import TCLaaCPipeline
pipeline = TCLaaCPipeline(num_topics=11)
pipeline.load_data('data.csv')
pipeline.preprocess()
pipeline.prepare_corpus()
pipeline.tune_hyperparameters()  # Finds optimal topic count
pipeline.train_model()
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
