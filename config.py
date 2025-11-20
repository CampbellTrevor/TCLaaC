"""
Centralized configuration for TCLaaC project.

This module contains all configurable parameters for the command-line analysis
pipeline, including model hyperparameters, file paths, and preprocessing rules.
"""

import re
import os

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Number of latent topics to discover in the LDA model
NUM_TOPICS = 11

# Random state for reproducibility across all operations
RANDOM_STATE = 42

# LDA Hyperparameters (optimized through tuning)
LDA_ALPHA = 'asymmetric'  # Document-topic prior
LDA_ETA = 1  # Topic-word prior
LDA_PASSES = 10  # Number of passes through the corpus during training
LDA_ITERATIONS = 100  # Maximum number of iterations through the corpus
LDA_CHUNKSIZE = 2000  # Number of documents to process in each batch

# Minimum number of tokens required for a document to be included
MIN_DOC_LENGTH = 2

# =============================================================================
# FILE PATHS
# =============================================================================

# Path to LOLBAS repository (Living Off The Land Binaries and Scripts)
LOLBAS_REPO_PATH = './OSBinaries'

# Output file paths
MODEL_FILENAME = 'lda_model.joblib'
ANALYSIS_DATAFRAME_FILENAME = 'analysis_dataframe.parquet'
TOPIC_SUMMARY_FILENAME = 'topic_summary.csv'

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Number of CPU cores to use (-1 = use all available cores)
N_JOBS = -1

# Automatically determine number of workers for Gensim LdaMulticore
NUM_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 1

# Maximum sample size for hyperparameter tuning (to speed up tuning)
MAX_TUNING_SAMPLE = 100000

# =============================================================================
# HYPERPARAMETER TUNING SETTINGS
# =============================================================================

# Range for number of topics during tuning
TUNING_MIN_TOPICS = 5
TUNING_MAX_TOPICS = 50
TUNING_STEP = 2

# Coherence metric to use ('c_v', 'u_mass', 'c_uci', 'c_npmi')
COHERENCE_METHOD = 'c_v'

# Alpha and Eta options for grid search
ALPHA_OPTIONS = [0.01, 0.1, 1, 'asymmetric', 'auto']
ETA_OPTIONS = [0.01, 0.1, 1, 'symmetric', 'auto']

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Fuzzy matching threshold for grouping similar commands (0-100)
SIMILARITY_THRESHOLD = 40

# Number of top words to display per topic
N_TOP_WORDS = 10

# =============================================================================
# NORMALIZATION RULES
# =============================================================================

# Pre-compiled regex patterns for command-line normalization
# These rules replace high-variability components with generic placeholders
# to help the model focus on command structure rather than specific values

NORMALIZATION_RULES_COMPILED = {
    # --- Identifier Rules (simple replacement) ---
    # Matches globally unique identifiers (GUIDs).
    'guids': (
        re.compile(r'\{?[a-f0-9]{8}-([a-f0-9]{4}-){3}[a-f0-9]{12}\}?', re.IGNORECASE),
        '<GUID>'
    ),
    # Matches long hexadecimal strings (32 characters or more).
    'long_hex_strings': (
        re.compile(r'\b[a-f0-9]{32,}\b', re.IGNORECASE),
        '<LONG_HEX_STRING>'
    ),
    # Matches long strings that resemble Base64-encoded data.
    'base64_strings': (
        re.compile(r'\b[A-Za-z0-9+/=]{50,}\b'),
        '<BASE64_STRING>'
    ),
    
    # --- File Path Rules (must come before network rules) ---
    # Matches UNC paths (before IP addresses to avoid conflicts)
    'unc_paths': (
        re.compile(r'\\\\[^\s<>:"/|?*]+\\[^\s<>:"/|?*\\]+'),
        '<UNC_PATH>'
    ),
    # Matches Windows paths with drive letters
    'windows_paths': (
        re.compile(r'[A-Za-z]:\\(?:[^\s<>:"/\\|?*]+\\)*[^\s<>:"/\\|?*]*'),
        '<WINDOWS_PATH>'
    ),
    
    # --- Network Rules ---
    # Matches URLs (http/https/ftp)
    'urls': (
        re.compile(r'\b(?:https?|ftp)://[^\s<>"{}|\\^`\[\]]+\b', re.IGNORECASE),
        '<URL>'
    ),
    # Matches IPv4 addresses, with an optional port number.
    'ip_addresses': (
        re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?\b'),
        '<IP_ADDRESS>'
    ),
    # Matches email addresses
    'emails': (
        re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
        '<EMAIL>'
    ),
    # Matches domain names without protocol
    'domains': (
        re.compile(r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b', re.IGNORECASE),
        '<DOMAIN>'
    ),

    # --- Date & Time Rules ---
    # Matches dates in YYYY-MM-DD format.
    'dates_yyyy-mm-dd': (
        re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
        '<DATE>'
    ),
    # Matches dates in MM/DD/YYYY format
    'dates_mm-dd-yyyy': (
        re.compile(r'\b\d{2}/\d{2}/\d{4}\b'),
        '<DATE>'
    ),
    # Matches times in HH:MM:SS format.
    'times_hh-mm-ss': (
        re.compile(r'\b\d{2}:\d{2}:\d{2}\b'),
        '<TIME>'
    ),
    # Matches timestamps with milliseconds
    'timestamps': (
        re.compile(r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3,6})?Z?\b'),
        '<TIMESTAMP>'
    ),

    # --- Windows Registry Rules ---
    # Matches Windows registry paths
    'registry_keys': (
        re.compile(r'\b(?:HKEY_[A-Z_]+|HKLM|HKCU|HKCR|HKU|HKCC)(?:\\[^\s\\]+)*\b', re.IGNORECASE),
        '<REGISTRY_KEY>'
    ),

    # --- General Number & Code Rules ---
    # Matches hexadecimal codes (e.g., 0xdeadbeef).
    'hex_codes': (
        re.compile(r'\b0x[a-f0-9]+\b', re.IGNORECASE),
        '<HEX_CODE>'
    ),
    # Matches long sequences of digits (6 or more).
    'long_numbers': (
        re.compile(r'\b\d{6,}\b'),
        '<LONG_NUMBER>'
    ),
    # Matches process IDs (PID: followed by numbers)
    'process_ids': (
        re.compile(r'\bPID:\s*\d+\b', re.IGNORECASE),
        '<PROCESS_ID>'
    )
}

# =============================================================================
# EXECUTABLE FILE EXTENSIONS
# =============================================================================

# Set of known executable file extensions for identifying root commands
EXECUTABLE_EXTENSIONS = {
    '.exe', '.com', '.bat', '.cmd', '.ps1', 
    '.scr', '.cpl', '.msc', '.vbs', '.js'
}

# =============================================================================
# DATA GENERATION SETTINGS (for synthetic data)
# =============================================================================

# Common Windows executables for synthetic data generation
COMMON_EXECUTABLES = [
    'powershell.exe', 'cmd.exe', 'net.exe', 'wmic.exe',
    'reg.exe', 'rundll32.exe', 'regsvr32.exe', 'mshta.exe',
    'certutil.exe', 'bitsadmin.exe', 'schtasks.exe', 'sc.exe'
]

# Common command patterns for synthetic data
COMMON_PATTERNS = [
    '{exe} /c whoami',
    '{exe} -ExecutionPolicy Bypass -File {file}',
    '{exe} query user',
    '{exe} add value',
    '{exe} /create /tn {task} /tr {path}',
]

# =============================================================================
# SECURITY ANALYSIS CONFIGURATION
# =============================================================================

# MITRE ATT&CK technique mapping keywords
MITRE_ATTACK_PATTERNS = {
    'T1059': ['powershell', 'cmd', 'wscript', 'cscript', 'bash'],  # Command and Scripting Interpreter
    'T1053': ['schtasks', 'at.exe', 'cron'],  # Scheduled Task/Job
    'T1105': ['certutil', 'bitsadmin', 'wget', 'curl', 'download'],  # Ingress Tool Transfer
    'T1218': ['rundll32', 'regsvr32', 'mshta', 'msiexec'],  # System Binary Proxy Execution
    'T1547': ['reg add', 'hklm\\software\\microsoft\\windows\\currentversion\\run'],  # Boot or Logon Autostart
    'T1003': ['lsass', 'mimikatz', 'procdump', 'dump'],  # Credential Dumping
    'T1055': ['inject', 'createremotethread'],  # Process Injection
    'T1027': ['base64', 'encode', 'compress', 'obfuscate'],  # Obfuscated Files or Information
}

# Command sequence patterns that indicate attack chains
ATTACK_CHAIN_PATTERNS = [
    ['powershell', 'download', 'execute'],
    ['certutil', 'decode', 'execute'],
    ['reg add', 'run'],
    ['schtasks', 'create'],
]

# Risk score weights
RISK_SCORE_WEIGHTS = {
    'lolbas_density': 0.4,
    'mitre_coverage': 0.3,
    'command_complexity': 0.15,
    'unique_binaries': 0.15,
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
LOG_LEVEL = 'INFO'

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """
    Validates configuration parameters to ensure they are within acceptable ranges.
    
    Raises:
        ValueError: If any configuration parameter is invalid.
    """
    if NUM_TOPICS < 2:
        raise ValueError(f"NUM_TOPICS must be >= 2, got {NUM_TOPICS}")
    
    if MIN_DOC_LENGTH < 1:
        raise ValueError(f"MIN_DOC_LENGTH must be >= 1, got {MIN_DOC_LENGTH}")
    
    if not (0 <= SIMILARITY_THRESHOLD <= 100):
        raise ValueError(f"SIMILARITY_THRESHOLD must be between 0 and 100, got {SIMILARITY_THRESHOLD}")
    
    if TUNING_MIN_TOPICS >= TUNING_MAX_TOPICS:
        raise ValueError("TUNING_MIN_TOPICS must be less than TUNING_MAX_TOPICS")
    
    if not os.path.exists(LOLBAS_REPO_PATH):
        import logging
        logging.getLogger(__name__).debug(f"LOLBAS repository not found at {LOLBAS_REPO_PATH}")

# Run validation on import
if __name__ != '__main__':
    validate_config()
