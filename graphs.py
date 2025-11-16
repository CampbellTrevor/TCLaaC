"""
A collection of utility functions for topic modeling, specifically for tuning
Latent Dirichlet Allocation (LDA) models and visualizing their results.

This module provides functions to:
- Tune the number of topics for an LDA model using perplexity scores.
- Generate interactive treemaps to visualize topic and command distributions.

It relies on libraries such as pandas, plotly, matplotlib, scikit-learn,
and gensim for data manipulation, visualization, and topic modeling.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import helpers
import re
import ipywidgets as widgets
from ipywidgets import VBox
from rapidfuzz import fuzz
from collections import defaultdict
from sklearn.decomposition import LatentDirichletAllocation,TruncatedSVD
from sklearn.preprocessing import normalize
import optuna
from sklearn.model_selection import cross_val_score
from joblib import parallel_backend
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel, LdaModel

def tune_lda_perplexity(doc_term_matrix: np.ndarray, min_topics: int = 5, max_topics: int = 50, step: int = 5):
    """
    Tunes the number of topics for an LDA model using perplexity and plots the results.

    This function iterates through a range of topic numbers, trains an LDA model
    for each, calculates the perplexity score, and then plots these scores.
    Perplexity is a measure of how well a probability model predicts a sample;
    in the context of LDA, a lower perplexity score indicates a better model fit.

    Args:
        doc_term_matrix (np.ndarray): The document-term matrix, typically generated
                                      by scikit-learn's CountVectorizer.
        min_topics (int): The minimum number of topics to test in the search range.
        max_topics (int): The maximum number of topics to test in the search range.
        step (int): The step size to use when iterating from min_topics to max_topics.

    Returns:
        None: This function does not return any value. It prints the results
              to the console and displays a matplotlib plot.
    """
    print("--- Tuning LDA with Perplexity Score ---")
    # Define the range of topic numbers to evaluate.
    topic_range = range(min_topics, max_topics + 1, step)
    perplexity_scores = []

    # Iterate over the specified range of topic counts. For each count,
    # train an LDA model and compute its perplexity on the provided data.
    # Perplexity measures how well the model generalizes to unseen data.
    for n_topics in topic_range:
        # Train an optimized LDA model for the current number of topics.
        lda = helpers.train_lda_optimized(doc_term_matrix, n_topics)

        # Calculate and store the perplexity score.
        perplexity_scores.append(lda.perplexity(doc_term_matrix))
        print(f"  Topics: {n_topics}, Perplexity: {perplexity_scores[-1]:.2f}")

    # After evaluating all topic numbers, visualize the results to help
    # identify the optimal number of topics (the "elbow point" where the
    # perplexity score begins to level off).
    plt.figure(figsize=(10, 6))
    plt.plot(topic_range, perplexity_scores, marker='o')
    plt.title("LDA Perplexity Score vs. Number of Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity Score (Lower is Better)")
    plt.xticks(topic_range)
    plt.grid(True)
    plt.show()

    # Identify and report the number of topics that yielded the lowest
    # perplexity score, as this is statistically the best model in the test.
    best_score_index = np.argmin(perplexity_scores)
    best_n_topics = topic_range[best_score_index]
    print(f"\nBest Perplexity Score found at {best_n_topics} topics.")


def create_topic_treemap_gensim(df_with_topics: pd.DataFrame, lda_model: LdaMulticore, similarity_threshold: int = 40) -> go.Figure:
    """
    Generates a grouped, interactive treemap from a gensim LDA model.

    This function performs three main steps:
    1. Extracts human-readable topic summaries from the trained gensim model.
    2. Groups similar command strings within the data using fuzzy string matching
       to create canonical command groups.
    3. Aggregates the data by topic and command group to generate a treemap
       visualization with Plotly, showing the prevalence of command groups
       within each topic.

    Args:
        df_with_topics (pd.DataFrame): DataFrame containing the source data, which
                                       must include a 'topic' column with topic
                                       assignments and a 'normalized_command'
                                       column with the command strings to be grouped.
        lda_model (LdaMulticore): The trained gensim LdaMulticore model from which
                                  to derive topic summaries.
        similarity_threshold (int): The RapidFuzz ratio score (0-100) above which
                                    two command strings are considered similar enough
                                    to be placed in the same group.

    Returns:
        go.Figure: A Plotly Figure object representing the interactive treemap.

    Raises:
        KeyError: If the input DataFrame `df_with_topics` does not contain the
                  required 'normalized_command' column.
    """
    print("--- Preparing data and generating treemap for gensim model ---")

    # Part 1: Generate topic summaries directly from the gensim model.
    # This process extracts the top keywords for each topic to create a
    # descriptive, human-readable label.
    # The result is a dictionary mapping topic indices to formatted strings.
    topic_summaries = {
        topic_idx: f"Topic {topic_idx}: " + ", ".join([word.split('*')[1].strip().strip('"') for word in topic.split(' + ')])
        for topic_idx, topic in lda_model.print_topics(num_words=4)
    }
    # Add a summary for documents that were not strongly assigned to any topic.
    topic_summaries[-1] = "Unassigned"

    # Map the generated summaries onto the DataFrame for use in the plot.
    df_with_topics['topic_summary'] = df_with_topics['topic'].map(topic_summaries)

    # Part 2: Group similar commands using an optimized fuzzy matching algorithm.
    # This step reduces the cardinality of command strings by clustering
    # variations of the same command into a single canonical representation.
    # This makes the final visualization cleaner and more insightful.
    print("Grouping similar commands using fuzzy matching...")
    template_groups = {}

    # The 'normalized_command' column is essential for this grouping step.
    if 'normalized_command' not in df_with_topics.columns:
        raise KeyError("DataFrame must contain a 'normalized_command' column for grouping.")

    unique_templates = df_with_topics['normalized_command'].unique()

    # To optimize the fuzzy matching process, first bucket the unique commands
    # by their root (e.g., 'git clone' and 'git push' both go into the 'git'
    # bucket). This drastically reduces the number of pairwise comparisons needed.
    template_buckets = defaultdict(list)
    for template in unique_templates:
        key = helpers.identify_root(template)
        template_buckets[key].append(template)

    # Now, run the fuzzy matching algorithm only within each bucket. For each
    # template, compare it against the identified canonical representatives. If a
    # match is found above the similarity threshold, group it; otherwise, it
    # becomes a new canonical representative for its group.
    for bucket in template_buckets.values():
        canonical_reps = []
        for template in bucket:
            is_grouped = False
            for rep in canonical_reps:
                if fuzz.ratio(template, rep) > similarity_threshold:
                    template_groups[template] = rep
                    is_grouped = True
                    break
            if not is_grouped:
                template_groups[template] = template
                canonical_reps.append(template)

    # Map the group assignments back to the main DataFrame.
    df_with_topics['command_group'] = df_with_topics['normalized_command'].map(template_groups)
    print("Grouping complete.")

    # Part 3: Aggregate the data and generate the treemap plot.
    # The data is grouped by topic summary and the newly created command
    # groups, and the size of each group is counted.
    treemap_data = df_with_topics.groupby(['topic_summary', 'command_group']).size().reset_index(name='count')

    # Create the treemap using Plotly Express. The hierarchy is defined
    # by topic, then by command group. The size of each rectangle in the
    # treemap corresponds to the count of commands in that group.
    fig = px.treemap(
        treemap_data,
        path=[px.Constant("All Topics"), 'topic_summary', 'command_group'],
        values='count',
        color='topic_summary',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title='Grouped Treemap of Command Line Topics',
        hover_data={'count': ':d'}
    )
    # Adjust layout for better readability.
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), font=dict(size=14))

    return fig