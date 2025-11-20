"""
A suite of helper functions for processing and analyzing command-line data.

This module provides utilities for parsing, normalizing, and tokenizing raw
command-line strings. It also includes functions for topic modeling with
Latent Dirichlet Allocation (LDA), identifying anomalous logs based on term
frequency, and analyzing topics for potentially malicious activity using
keywords derived from security resources like the LOLBAS project.
"""

import os
import pandas as pd
from rapidfuzz import fuzz
import re
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel, LdaModel
import time
import shlex
import ntpath
import yaml
import glob
import logging

logger = logging.getLogger(__name__)

# A dictionary of compiled regular expression rules used for normalizing
# command-line strings. Each key represents a category of patterns to be
# replaced, and the value is a tuple containing the compiled regex pattern
# and the placeholder string to substitute. This pre-compilation improves
# performance when applying the rules repeatedly.
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
    
    # --- Network Rules ---
    # Matches IPv4 addresses, with an optional port number.
    'ip_addresses': (
        re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?\b'),
        '<IP_ADDRESS>'
    ),

    # --- Date & Time Rules ---
    # Matches dates in YYYY-MM-DD format.
    'dates_yyyy-mm-dd': (
        re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
        '<DATE>'
    ),
    # Matches times in HH:MM:SS format.
    'times_hh-mm-ss': (
        re.compile(r'\b\d{2}:\d{2}:\d{2}\b'),
        '<TIME>'
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
    )
}

def normalize_command(command_str: str, rules_dict: dict, max_loops: int = 15) -> str:
    """
    Iteratively normalizes a command string by applying a set of regex rules.

    The function repeatedly applies all rules in the provided dictionary to the
    command string until a full pass over the rules results in no changes to
    the string. This iterative approach handles nested or overlapping patterns.
    A loop limit is included as a safeguard against infinite loops.

    Args:
        command_str (str): The raw command-line string to be normalized.
        rules_dict (dict): A dictionary where keys are rule names and values
                           are tuples of (compiled_regex, replacement_string).
        max_loops (int): The maximum number of iterations to prevent infinite loops.

    Returns:
        str: The normalized command string. Returns an empty string if the
             input is not a valid string.
    """
    # Ensure the input is a string before proceeding.
    if not isinstance(command_str, str):
        return ""

    # A safety break is included to prevent potential infinite loops in case
    # a rule inadvertently creates a pattern that a subsequent rule matches,
    # leading to a cycle.
    loops = 0
    while loops < max_loops:
        previous_state = command_str

        # Apply each rule in the dictionary to the current command string.
        for rule_name, (pattern, replacement) in rules_dict.items():
            command_str = pattern.sub(replacement, command_str)
        
        # If the string has not changed after a full pass through all the rules,
        # the normalization process is complete, and we can exit the loop.
        if command_str == previous_state:
            break
            
        loops += 1
    
    # Warn if max loops reached (potential normalization issue)
    if loops >= max_loops:
        logger.warning(f"Normalization reached max loops ({max_loops}) for command: {command_str[:100]}...")
        
    return command_str

def extract_command_line(message: str = None) -> str | None:
    """
    Extracts a command line string from a larger text block.

    This function searches for a line explicitly prefixed with 'CommandLine:'
    within the provided message and returns the content that follows.

    Args:
        message (str, optional): The input text containing the command line.
            Defaults to None.

    Returns:
        str | None: The extracted command line string if found, otherwise None.
    """
    # Ensure newline characters are consistently represented as literal '\\n'.
    # This standardizes input that might use different newline conventions
    # (\n, \r\n) before the regex search is performed.
    if '\\n' not in message:
        message = message.replace('\n', '\\n').replace('\r', '')

    # Search for the pattern '\\nCommandLine:' followed by any characters
    # (captured non-greedily) until the next '\\n'. The re.DOTALL flag
    # is included for robustness, allowing '.' to match newline characters
    # if they appear within the command line itself.
    match = re.search(r'\\nCommandLine:\s*(.*?)\\n', message, re.DOTALL)

    # If a match object was returned, extract the first captured group,
    # which contains the command line text. Otherwise, return None.
    return match.group(1) if match else None

def tokenize(cmd: str) -> list[str]:
    """
    Splits a command line string into a list of logical tokens.

    This function uses a comprehensive regular expression to identify various
    components of a command line, such as quoted strings, URLs, file paths,
    flags, and other common patterns. It handles complex cases that a
    simple split on whitespace would miss.

    Args:
        cmd (str): The command line string to be tokenized.

    Returns:
        list[str]: A list of tokens extracted from the command string. Returns
                   an empty list if the input is None.
    """
    # If the input command is None, return an empty list immediately to
    # prevent errors in subsequent processing.
    if isinstance(cmd, type(None)):
        return []

    # This comprehensive regex is designed to capture distinct parts of a
    # command line as whole tokens. The re.VERBOSE flag allows the pattern
    # to be formatted with comments for readability. The patterns are ordered
    # from more specific (e.g., quoted strings, URLs) to more general
    # (e.g., single words) to ensure correct matching.
    token_pattern = r'''
    (?:<[a-zA-Z_]+>)                      # Normalized token
    | (?:(?!\b)"[^"]+"|'[^']+')           # Quoted Strings
    | (?:https?://[^\s"']+)               # Full URL
    | (?:--?[\w\-]+=[^\s"']+)             # --flag=value
    | (?:[a-zA-Z]:\\(?:[\w\-.\\ ]+))       # Windows paths
    | (?:/[\w\-/\.]+)                     # Unix paths
    | (?:\d{1,3}(?:\.\d{1,3}){3})         # IP Addresses
    | (?:--?[\w\-]+)                      # Arguments like -n, --urlcache
    | (?:[\w\-.]+\.(?:exe|dll|bat|ps1|txt|sh))# Filenames
    | (?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}) # Email addresses
    | (?:0x[a-fA-F0-9]+)                  # Hexadecimal numbers
    | (?:\d{4}-\d{2}-\d{2})                # Dates (YYYY-MM-DD)
    | (?:\d{2}/\d{2}/\d{4})                # Dates (MM/DD/YYYY)
    | (?:\d{2}:\d{2}:\d{2})                # Times (HH:MM:SS)
    | (?:\d{2}:\d{2} (?:AM|PM))            # Times (HH:MM AM/PM)
    | (?:[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}) # UUIDs
    | (?:[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})# MAC Addresses
    | (?:\d{3}\d{3} \d{3}-\d{4})            # Phone numbers (123) 456-7890
    | (?:\+\d{1,2}-\d{3}-\d{3}-\d{4})      # Phone numbers +1-800-555-5555
    | (?:[\w]+)                           # Fallback: single words
    '''

    # Find all non-overlapping matches of the token_pattern in the lowercase
    # version of the command. A list comprehension then strips any leading or
    # trailing quotes from each found token.
    return [t.strip('"\'') for t in re.findall(token_pattern, cmd.lower(), re.VERBOSE)]

# A set of common executable file extensions, used to identify the root
# command in a command line string.
EXECUTABLE_EXTENSIONS = {'.exe', '.com', '.bat', '.cmd', '.ps1', '.scr', '.cpl'}

def identify_root(cmd_string: str) -> str:
    """
    Identifies the root command from a complete command line string.

    The root command is defined as the executable file's name. The function
    parses the command line to find the token ending with a known executable
    extension and returns its base name. If no such token is found, the first
    token is assumed to be the root.

    Args:
        cmd_string (str): A command line string.

    Returns:
        str: The identified root command (e.g., 'powershell.exe'). Returns an
             empty string if the input is invalid or empty.
    """
    # Ensure the input is a processable string; otherwise, return an empty string.
    if not isinstance(cmd_string, str):
        return ""

    # Use shlex.split for robust, shell-aware parsing of the command line.
    # It correctly handles quotes and escaped characters. The posix=False
    # argument ensures it handles Windows-style paths and conventions.
    try:
        parts = shlex.split(cmd_string, posix=False)
    except ValueError:
        # If shlex fails (e.g., due to unmatched quotes), fall back to a
        # simple split by whitespace as a robust alternative.
        parts = cmd_string.split()
        
    if not parts:
        return ""

    # Iterate through the parsed parts of the command to find the first token
    # that appears to be an executable file based on its extension.
    root_command_end_index = -1
    for i, part in enumerate(parts):
        if any(part.lower().endswith(ext) for ext in EXECUTABLE_EXTENSIONS):
            root_command_end_index = i
            break
            
    # If an executable was found, construct the full path to it. If not,
    # assume the very first part of the command is the root.
    if root_command_end_index != -1:
        full_root_path = " ".join(parts[:root_command_end_index + 1])
    else:
        full_root_path = parts[0]
    
    # Extract just the filename from the full path to serve as the root command.
    executable_name = ntpath.basename(full_root_path.strip('"'))

    return executable_name


def isolate_anomalous_logs(df: pd.DataFrame, text_column: str, min_df_threshold: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into normal and anomalous logs based on term frequency.

    This method operates on the principle that anomalous events often involve
    rare commands or parameters. It identifies all terms (tokens) that appear
    in fewer documents than the specified `min_df_threshold`. Any log entry
    containing one or more of these rare terms is flagged as an anomaly.

    Args:
        df (pd.DataFrame): The input DataFrame containing the text data.
        text_column (str): The name of the column with the command strings.
        min_df_threshold (int): The document frequency below which a term is
                                considered rare/anomalous.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                                           (df_normal, df_anomalies).
    """
    print(f"--- Isolating anomalies using a minimum document frequency of {min_df_threshold} ---")
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Step 1: Vectorize the entire dataset to get term counts for each document.
    # The token pattern ensures that any non-whitespace sequence is a token.
    vectorizer = CountVectorizer(lowercase=False, token_pattern=r'[^\s]+')
    doc_term_matrix = vectorizer.fit_transform(df[text_column])

    # Step 2: Calculate the document frequency for each term. This is the
    # number of documents in which each term appears at least once.
    doc_frequencies = np.asarray((doc_term_matrix > 0).sum(axis=0)).flatten()

    # Step 3: Identify the vocabulary terms that are "rare" by comparing their
    # document frequency against the threshold.
    vocab = vectorizer.get_feature_names_out()
    rare_term_indices = np.where(doc_frequencies < min_df_threshold)[0]
    
    if len(rare_term_indices) == 0:
        print("No rare terms found based on the threshold.")
        return df, pd.DataFrame() # Return the original df and an empty df

    # Step 4: Create a boolean mask to identify documents (rows) that contain
    # at least one of the identified rare terms.
    # We slice the matrix to only include the columns for rare terms.
    rare_term_matrix = doc_term_matrix[:, rare_term_indices]
    
    # A document is considered anomalous if its row in the rare term matrix
    # sums to a value greater than 0.
    anomaly_mask = np.asarray(rare_term_matrix.sum(axis=1) > 0).flatten()

    # Step 5: Split the original DataFrame into two separate DataFrames using
    # the generated anomaly mask.
    df_anomalies = df[anomaly_mask]
    df_normal = df[~anomaly_mask]

    print(f"Found {len(df_anomalies)} logs containing rare terms.")
    print(f"Remaining {len(df_normal)} logs for topic modeling.")
    
    return df_normal, df_anomalies


def train_lda_optimized(doc_term_matrix, num_topics: int, random_state: int = 42, n_jobs: int = -1, doc_topic_prior=None, topic_word_prior=None) -> LatentDirichletAllocation:
    """
    Trains an optimized Latent Dirichlet Allocation (LDA) model.

    This function configures and trains an LDA model from scikit-learn using
    the specified document-term matrix and model parameters.

    Args:
        doc_term_matrix (scipy.sparse.csr_matrix | np.ndarray): The
            document-term matrix, where rows are documents and columns are
            unique terms.
        num_topics (int): The number of latent topics to discover in the corpus.
        random_state (int, optional): A seed for the random number generator
            to ensure reproducibility of results. Defaults to 42.
        n_jobs (int, optional): The number of parallel jobs to run. -1 means
            using all available processors. Defaults to -1.
        doc_topic_prior (float, optional): The hyperparameter alpha, which
            controls the sparsity of the document-topic distribution. Defaults to None.
        topic_word_prior (float, optional): The hyperparameter eta, which
            controls the sparsity of the topic-word distribution. Defaults to None.

    Returns:
        LatentDirichletAllocation: The trained LDA model object.
    """
    print(f"\nStep 2: Training LDA model with {num_topics} topics using batch learning...")

    start_time = time.time()
    # Instantiate the LDA model. The 'batch' learning method uses all training
    # data in each update step, which is computationally intensive but can
    # lead to more accurate models compared to 'online' learning.
    lda = LatentDirichletAllocation(n_components=num_topics,
                                      learning_method='batch',
                                      random_state=random_state,
                                      doc_topic_prior=doc_topic_prior,
                                      topic_word_prior=topic_word_prior,
                                      n_jobs=n_jobs)

    # Fit the LDA model to the document-term matrix.
    lda.fit(doc_term_matrix)

    end_time = time.time()
    print(f"Training complete in {end_time - start_time:.2f} seconds.")
    return lda

def get_topic_results_gensim(lda_model: LdaMulticore, corpus: list, df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Processes and appends LDA topic assignments from a gensim model to a DataFrame.

    This function assigns the most likely topic to each document in the original DataFrame.

    Args:
        lda_model (LdaMulticore): The trained gensim LDA model.
        corpus (list): The gensim corpus (list of bag-of-words) used for training.
        df_original (pd.DataFrame): The original DataFrame to which topic
                                    assignments will be added. Note: This should
                                    be the DataFrame that corresponds to the
                                    filtered_tokens used to create the corpus.

    Returns:
        pd.DataFrame: A copy of the original DataFrame with a new 'topic'
                      column containing the assigned topic index for each row.
                      
    Raises:
        ValueError: If the number of rows in `df_original` does not match the
                    number of documents in the `corpus`.
    """
    # Assign the single most likely topic to each document.
    topic_assignments = []
    for doc_bow in corpus:
        # get_document_topics returns a list of (topic_id, probability) tuples.
        doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)
        if not doc_topics:
            # If a document has no assigned topics (e.g., it was empty),
            # assign a default topic of -1 to indicate it's unclassified.
            topic_assignments.append(-1)
        else:
            # Find the topic with the highest probability for the current document.
            most_likely_topic = max(doc_topics, key=lambda x: x[1])
            topic_assignments.append(most_likely_topic[0])

    # A critical sanity check to prevent misaligned data. The number of documents
    # processed must match the number of rows in the DataFrame to be annotated.
    if len(df_original) != len(corpus):
        raise ValueError(
            f"The DataFrame length ({len(df_original)}) does not match the corpus length ({len(corpus)}). "
            "Please ensure you are using the correctly filtered DataFrame that corresponds to your final corpus."
        )

    # Create a copy of the original DataFrame to avoid modifying it in place.
    df_result = df_original.copy()
    # Add the topic assignments as a new column. The list of assignments
    # directly corresponds to the order of documents in the corpus and DataFrame.
    df_result['topic'] = topic_assignments

    return df_result

def generate_tfidf_summaries(lda_model: LatentDirichletAllocation, doc_term_matrix: np.ndarray, vectorizer, df_with_topics: pd.DataFrame, n_top_words: int = 10) -> dict:
    """
    Generates a data-driven summary for each topic using TF-IDF.

    This function creates a concise, human-readable label for each topic by
    identifying its most significant keyword based on TF-IDF scores, and then
    combining it with the most common executable found in that topic for context.

    Args:
        lda_model (LatentDirichletAllocation): The trained LDA model.
        doc_term_matrix (np.ndarray): The document-term matrix.
        vectorizer (CountVectorizer): The fitted vectorizer object.
        df_with_topics (pd.DataFrame): DataFrame with 'topic' and 'tokens' columns.
        n_top_words (int, optional): Number of top words per topic to consider
            for TF-IDF calculation. Defaults to 10.

    Returns:
        dict: A dictionary mapping each topic index to its generated summary string.
    """
    print("--- Generating TF-IDF Based Smart Summaries ---")
    summaries = {}
    feature_names = vectorizer.get_feature_names_out()

    # Step 1: Calculate Inverse Document Frequency (IDF) for all words.
    # IDF measures how informative a word is, giving higher weight to words
    # that appear in fewer documents.
    num_documents = doc_term_matrix.shape[0]
    # Count how many documents each word appears in.
    doc_freq = np.bincount(doc_term_matrix.indices, minlength=doc_term_matrix.shape[1])
    # The IDF formula; +1 is added for smoothing to avoid division by zero.
    idf = np.log(num_documents / (1 + doc_freq))

    # Step 2: Loop through each topic to find its most significant word.
    for topic_idx, topic_component in enumerate(lda_model.components_):
        # The topic component represents word counts. Normalizing it gives
        # word probabilities, which act as the Term Frequency (TF) for the topic.
        tf = topic_component / topic_component.sum()

        # Get the indices of the top N most probable words for this topic.
        top_word_indices = topic_component.argsort()[:-n_top_words - 1:-1]

        # Calculate the TF-IDF score for each of these top words. TF-IDF
        # balances a word's frequency within a topic (TF) with its rarity
        # across all documents (IDF), highlighting words that are important
        # specifically to this topic.
        top_tfidf_scores = tf[top_word_indices] * idf[top_word_indices]

        # The most significant word is the one with the highest TF-IDF score.
        most_significant_word_index = top_word_indices[np.argmax(top_tfidf_scores)]
        most_significant_word = feature_names[most_significant_word_index]

        # Step 3: Find the most common executable in the topic for context.
        # This helps ground the abstract topic in a concrete program name,
        # making the summary more interpretable.
        topic_docs = df_with_topics[df_with_topics['topic'] == topic_idx]

        # Gather all tokens from all documents belonging to the current topic.
        all_tokens = [token.lower() for sublist in topic_docs['tokens'] for token in sublist]
        executables = [token for token in all_tokens if token.endswith('.exe')]

        most_common_exe = ""
        if executables:
            # Find the most frequently occurring executable.
            most_common_exe = pd.Series(executables).mode()[0]
            # Clean up the name for use in the summary.
            most_common_exe = most_common_exe.split('\\')[-1].replace('.exe', '').capitalize()

        # Step 4: Assemble the final summary string.
        summary_base = most_significant_word.capitalize()

        # Combine the significant word and the common executable intelligently
        # to create a descriptive and non-redundant summary.
        if most_common_exe and most_common_exe.lower() in summary_base.lower():
            # If the exe name is the summary word, avoid redundancy.
            final_summary = f"{summary_base} Activity"
        elif most_common_exe:
            # Otherwise, combine them for a more descriptive summary.
            final_summary = f"{most_common_exe} ({summary_base})"
        else:
            # If no common exe, just use the significant word.
            final_summary = f"Topic: {summary_base}"

        summaries[topic_idx] = final_summary

    return summaries

def map_mitre_attack_techniques(command: str, mitre_patterns: dict = None) -> list:
    """
    Maps a command to MITRE ATT&CK techniques based on pattern matching.
    
    Args:
        command: Command line string
        mitre_patterns: Dictionary of technique IDs to keyword patterns
        
    Returns:
        List of MITRE ATT&CK technique IDs that match the command
    """
    if mitre_patterns is None:
        # Import here to avoid circular dependency
        try:
            from config import MITRE_ATTACK_PATTERNS
            mitre_patterns = MITRE_ATTACK_PATTERNS
        except ImportError:
            return []
    
    command_lower = command.lower()
    matched_techniques = []
    
    for technique_id, keywords in mitre_patterns.items():
        if any(keyword.lower() in command_lower for keyword in keywords):
            matched_techniques.append(technique_id)
    
    return matched_techniques


def calculate_command_complexity(command: str) -> float:
    """
    Calculates a complexity score for a command based on multiple factors.
    
    Args:
        command: Command line string
        
    Returns:
        Complexity score (0-100)
    """
    score = 0.0
    
    # Factor 1: Command length (max 20 points)
    length_score = min(len(command) / 10, 20)
    score += length_score
    
    # Factor 2: Number of special characters (max 20 points)
    special_chars = sum(1 for c in command if c in '|&;<>(){}[]$`\'"')
    special_score = min(special_chars * 2, 20)
    score += special_score
    
    # Factor 3: Obfuscation indicators (max 30 points)
    obfuscation_keywords = ['base64', 'encode', 'hidden', 'bypass', 'noprofile']
    obfuscation_score = sum(10 for kw in obfuscation_keywords if kw.lower() in command.lower())
    score += min(obfuscation_score, 30)
    
    # Factor 4: Nesting level (pipes, redirects) (max 30 points)
    nesting_chars = command.count('|') + command.count('&&') + command.count(';')
    nesting_score = min(nesting_chars * 10, 30)
    score += nesting_score
    
    return min(score, 100)


def analyze_malicious_topics(df_with_topics: pd.DataFrame, suspicious_keywords: list) -> pd.DataFrame:
    """
    Scores topics based on LOLBAS density and command patterns.

    This function analyzes a DataFrame of documents with assigned topics. It
    calculates a "LOLBAS density" score for each topic based on multiple factors:
    - Percentage of documents containing LOLBAS binaries
    - Number of unique LOLBAS binaries in the topic
    - Average number of LOLBAS matches per document
    
    This provides a more nuanced security risk assessment than simple keyword matching.

    Args:
        df_with_topics (pd.DataFrame): DataFrame containing at least 'topic'
            and 'command_line' columns.
        suspicious_keywords (list): A list of strings representing LOLBAS
            binaries to search for.

    Returns:
        pd.DataFrame: A DataFrame indexed by topic ID, containing counts,
            LOLBAS density scores, and detailed keyword breakdowns.
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing topics using LOLBAS density metrics...")

    # Step 1: Flag individual commands that contain any LOLBAS binary.
    # Each keyword is escaped to ensure special characters (e.g., '.', '*')
    # are treated as literals, preventing regex errors or misinterpretations.
    escaped_keywords = [re.escape(kw) for kw in suspicious_keywords]
    pattern = r'\b(' + '|'.join(escaped_keywords) + r')\b'
    
    df_with_topics['contains_lolbas'] = df_with_topics['command_line'].str.contains(
        pattern, 
        regex=True,
        na=False
    )
    
    # Count LOLBAS matches per command
    df_with_topics['lolbas_count'] = df_with_topics['command_line'].str.findall(pattern).apply(len)

    # Add MITRE ATT&CK technique mapping
    try:
        from config import MITRE_ATTACK_PATTERNS
        df_with_topics['mitre_techniques'] = df_with_topics['command_line'].apply(
            lambda cmd: map_mitre_attack_techniques(cmd, MITRE_ATTACK_PATTERNS)
        )
        df_with_topics['mitre_count'] = df_with_topics['mitre_techniques'].apply(len)
        has_mitre = True
    except Exception as e:
        logger.warning(f"Could not map MITRE ATT&CK techniques: {e}")
        has_mitre = False
    
    # Add command complexity scoring
    df_with_topics['complexity_score'] = df_with_topics['command_line'].apply(calculate_command_complexity)
    
    # Step 2: Calculate comprehensive metrics per topic
    topic_metrics = []
    
    for topic_id in df_with_topics['topic'].unique():
        topic_data = df_with_topics[df_with_topics['topic'] == topic_id]
        total_docs = len(topic_data)
        lolbas_docs = topic_data['contains_lolbas'].sum()
        
        # Calculate density metrics
        lolbas_percentage = (lolbas_docs / total_docs) * 100 if total_docs > 0 else 0
        
        # Count unique LOLBAS binaries in this topic
        if lolbas_docs > 0:
            all_matches = topic_data[topic_data['contains_lolbas']]['command_line'].str.findall(pattern).sum()
            unique_lolbas = len(set(all_matches))
            avg_lolbas_per_doc = topic_data['lolbas_count'].sum() / total_docs
        else:
            unique_lolbas = 0
            avg_lolbas_per_doc = 0
        
        # Calculate LOLBAS density score (weighted metric)
        # Formula: (percentage * 0.5) + (unique_binaries * 5) + (avg_per_doc * 10)
        # This balances prevalence, diversity, and intensity of LOLBAS usage
        lolbas_density = (lolbas_percentage * 0.5) + (unique_lolbas * 5) + (avg_lolbas_per_doc * 10)
        
        # Calculate MITRE ATT&CK coverage
        if has_mitre:
            mitre_docs = (topic_data['mitre_count'] > 0).sum()
            mitre_percentage = (mitre_docs / total_docs) * 100 if total_docs > 0 else 0
            unique_techniques = set()
            for techniques in topic_data['mitre_techniques']:
                unique_techniques.update(techniques)
            mitre_coverage = len(unique_techniques)
        else:
            mitre_percentage = 0
            mitre_coverage = 0
        
        # Calculate average complexity
        avg_complexity = topic_data['complexity_score'].mean()
        
        # Calculate comprehensive risk score
        risk_score = (
            lolbas_density * 0.4 +
            mitre_coverage * 10 * 0.3 +
            avg_complexity * 0.15 +
            unique_lolbas * 2 * 0.15
        )
        
        topic_metrics.append({
            'topic': topic_id,
            'total_count': total_docs,
            'lolbas_count': int(lolbas_docs),
            'lolbas_percentage': lolbas_percentage,
            'unique_lolbas': unique_lolbas,
            'avg_lolbas_per_doc': avg_lolbas_per_doc,
            'lolbas_density': lolbas_density,
            'mitre_technique_count': mitre_coverage if has_mitre else 0,
            'mitre_percentage': mitre_percentage if has_mitre else 0,
            'avg_complexity': avg_complexity,
            'risk_score': risk_score
        })
    
    topic_scores = pd.DataFrame(topic_metrics).set_index('topic')
    topic_scores = topic_scores.sort_values(by='risk_score', ascending=False)

    # Step 3: Extract the specific binaries matched in each topic
    lolbas_df = df_with_topics[df_with_topics['contains_lolbas']].copy()
    lolbas_df['matched_lolbas'] = lolbas_df['command_line'].str.findall(pattern)
    
    # Step 4: Display summary of most suspicious topics
    logger.info("\n=== Top 5 Highest Risk Topics (by Comprehensive Risk Score) ===")
    for topic_id, row in topic_scores.head(5).iterrows():
        logger.info(f"\n🔴 Topic #{topic_id} | Overall Risk Score: {row['risk_score']:.2f}")
        logger.info(f"   LOLBAS Coverage: {row['lolbas_percentage']:.1f}% ({int(row['lolbas_count'])} of {int(row['total_count'])} docs)")
        logger.info(f"   LOLBAS Density: {row['lolbas_density']:.2f}")
        logger.info(f"   Unique LOLBAS Binaries: {row['unique_lolbas']}")
        
        if has_mitre:
            logger.info(f"   MITRE ATT&CK Techniques: {int(row['mitre_technique_count'])} unique")
            logger.info(f"   MITRE Coverage: {row['mitre_percentage']:.1f}% of docs")
        
        logger.info(f"   Avg Complexity Score: {row['avg_complexity']:.1f}/100")
        
        # Show most common LOLBAS binaries in this topic
        topic_lolbas = lolbas_df[lolbas_df['topic'] == topic_id]
        if len(topic_lolbas) > 0:
            all_binaries = [binary for binaries in topic_lolbas['matched_lolbas'] for binary in binaries]
            from collections import Counter
            top_binaries = Counter(all_binaries).most_common(5)
            logger.info("   Top LOLBAS Binaries:")
            for binary, count in top_binaries:
                logger.info(f"     • {binary}: {count}x")
        
        # Show MITRE techniques if available
        if has_mitre:
            topic_techniques = set()
            for techniques in df_with_topics[df_with_topics['topic'] == topic_id]['mitre_techniques']:
                topic_techniques.update(techniques)
            if topic_techniques:
                logger.info(f"   MITRE ATT&CK Techniques: {', '.join(sorted(topic_techniques))}")
    
    return topic_scores


def generate_keywords_from_lolbas(repo_path: str) -> list:
    """
    Parses YAML files from a local LOLBAS repository to extract keywords.

    This function iterates through all `.yml` files in the specified path,
    parsing each to extract the 'Name' of the binary and the executables from
    its associated 'Commands'. This creates a list of terms related to
    "Living Off The Land Binaries and Scripts".

    Args:
        repo_path (str): The local file path to the directory containing the
                         LOLBAS project YAML files.

    Returns:
        list: A sorted list of unique keywords (binary and command names).
    """
    keywords = set()
    # Define the search pattern for all YAML files within the specified directory.
    search_path = os.path.join(repo_path, '*.yml')

    for filepath in glob.glob(search_path):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
                
                # Add the primary binary name from the 'Name' field.
                if 'Name' in data:
                    keywords.add(data['Name'])
                
                # Extract executables from the example commands.
                if 'Commands' in data:
                    for command in data['Commands']:
                        if 'Command' in command and command['Command']:
                            parts = command['Command'].split()
                            if len(parts) > 0:
                                keywords.add(parts[0])
            except yaml.YAMLError as e:
                print(f"Error parsing {filepath}: {e}")

    return sorted(list(keywords))

def get_all_lolbas_commands(repo_path: str) -> list:
    """
    Parses the LOLBAS repository and extracts all full command-line examples.

    This function reads all `.yml` files in the given LOLBAS repository path
    and collects the full command strings listed under the 'Commands' section
    of each file.

    Args:
        repo_path (str): The local file path to the directory containing the
                         LOLBAS project YAML files.

    Returns:
        list: A list of all command-line strings found in the repository.
    """
    command_list = []
    # Define the path to the YAML files.
    search_path = os.path.join(repo_path, '*.yml')

    for filepath in glob.glob(search_path):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
                
                # Check if the 'Commands' key exists and is not empty.
                if 'Commands' in data and data['Commands'] is not None:
                    for command_entry in data['Commands']:
                        # Ensure the 'Command' key exists and contains a value.
                        if 'Command' in command_entry and command_entry['Command']:
                            command_list.append(command_entry['Command'])
            except yaml.YAMLError as e:
                print(f"Error parsing {filepath}: {e}")

    return command_list