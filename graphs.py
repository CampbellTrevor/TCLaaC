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
from collections import Counter

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


def generate_smart_topic_summary(topic_idx: int, lda_model: LdaMulticore, df_with_topics: pd.DataFrame) -> str:
    """
    Generates a smart, human-readable summary for a topic based on content analysis.
    
    Instead of just listing TF-IDF keywords, this analyzes the actual commands in the topic
    to identify the most common executable and the primary action/purpose.
    
    Args:
        topic_idx: The topic index
        lda_model: The trained LDA model
        df_with_topics: DataFrame with topic assignments and tokens
        
    Returns:
        A descriptive string summarizing what the topic represents
    """
    # Get top words for context
    topic_words = [word for word, _ in lda_model.show_topic(topic_idx, topn=10)]
    
    # Get documents in this topic
    topic_docs = df_with_topics[df_with_topics['topic'] == topic_idx]
    
    if len(topic_docs) == 0:
        return f"Topic {topic_idx}: Empty"
    
    # Analyze root commands
    if 'root_command' in topic_docs.columns:
        root_commands = topic_docs['root_command'].value_counts()
        most_common_exe = root_commands.index[0] if len(root_commands) > 0 else None
    else:
        most_common_exe = None
    
    # Analyze tokens to find common patterns
    all_tokens = []
    for tokens in topic_docs['tokens']:
        all_tokens.extend(tokens)
    
    token_counts = Counter(all_tokens)
    
    # Filter out the executable name itself to find action words
    filtered_tokens = {k: v for k, v in token_counts.most_common(15) 
                      if k not in [most_common_exe, most_common_exe.lower() if most_common_exe else None]}
    
    # Identify the nature of the topic based on key patterns
    action_indicators = {
        'network': ['http', 'url', 'download', 'web', 'net', 'tcp', 'ip', 'port'],
        'file': ['file', 'dir', 'copy', 'move', 'delete', 'write', 'read', 'path'],
        'registry': ['reg', 'hkey', 'hklm', 'hkcu', 'registry', 'key'],
        'execution': ['exec', 'run', 'start', 'invoke', 'call', 'execute', 'launch'],
        'scripting': ['script', 'powershell', 'cmd', 'batch', 'command'],
        'admin': ['admin', 'user', 'privilege', 'elevated', 'system'],
        'scheduled': ['task', 'schtasks', 'schedule', 'cron', 'at'],
        'encoding': ['encode', 'decode', 'base64', 'certutil'],
    }
    
    detected_category = None
    for category, keywords in action_indicators.items():
        if any(keyword in token for token in filtered_tokens.keys() for keyword in keywords):
            detected_category = category
            break
    
    # Build a descriptive name
    if most_common_exe:
        exe_name = most_common_exe.replace('.exe', '').capitalize()
        if detected_category:
            return f"{exe_name} {detected_category.capitalize()} Operations"
        else:
            # Use the most common non-exe token as descriptor
            descriptive_tokens = [t for t in filtered_tokens.keys() 
                                if not t.startswith('<') and len(t) > 2]
            if descriptive_tokens:
                descriptor = descriptive_tokens[0].capitalize()
                return f"{exe_name} {descriptor} Activity"
            return f"{exe_name} Commands"
    else:
        if detected_category:
            return f"{detected_category.capitalize()} Operations"
        # Fallback to top meaningful words
        meaningful = [w for w in topic_words if not w.startswith('<') and len(w) > 2][:3]
        return f"Topic {topic_idx}: {', '.join(meaningful).capitalize()}"


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
    print("Generating topic treemap visualization...")

    # Part 1: Generate smart topic summaries based on content analysis
    topic_summaries = {}
    for topic_idx in range(lda_model.num_topics):
        topic_summaries[topic_idx] = generate_smart_topic_summary(topic_idx, lda_model, df_with_topics)
    
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
    # OPTIMIZATION: Use process.extractOne for faster fuzzy matching
    from rapidfuzz import process
    
    for bucket in template_buckets.values():
        canonical_reps = []
        for template in bucket:
            if not canonical_reps:
                # First template in bucket becomes canonical
                template_groups[template] = template
                canonical_reps.append(template)
            else:
                # Use extractOne for O(n) instead of O(n²) comparison
                best_match = process.extractOne(
                    template, 
                    canonical_reps, 
                    scorer=fuzz.ratio,
                    score_cutoff=similarity_threshold
                )
                
                if best_match:
                    # Match found, group with canonical representative
                    template_groups[template] = best_match[0]
                else:
                    # No match, this becomes a new canonical rep
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


def create_topic_heatmap(df_with_topics: pd.DataFrame, lda_model: LdaMulticore) -> go.Figure:
    """
    Creates an interactive heatmap showing the distribution of top words across topics.
    
    Args:
        df_with_topics: DataFrame with topic assignments
        lda_model: Trained LDA model
        
    Returns:
        Plotly Figure object
    """
    # Get top words for each topic
    num_topics = lda_model.num_topics
    top_n = 15
    
    # Build matrix of word probabilities across topics
    topic_words = {}
    all_words = set()
    
    for topic_idx in range(num_topics):
        words_probs = lda_model.show_topic(topic_idx, topn=top_n)
        topic_words[topic_idx] = {word: prob for word, prob in words_probs}
        all_words.update([word for word, _ in words_probs])
    
    # Create matrix
    words_list = sorted(list(all_words))
    matrix = []
    for word in words_list:
        row = [topic_words[topic_idx].get(word, 0) for topic_idx in range(num_topics)]
        matrix.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f'Topic {i}' for i in range(num_topics)],
        y=words_list,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Topic-Word Distribution Heatmap',
        xaxis_title='Topics',
        yaxis_title='Words',
        height=max(600, len(words_list) * 20)
    )
    
    return fig


def create_security_score_chart(topic_scores: pd.DataFrame, df_with_topics: pd.DataFrame, lda_model: LdaMulticore) -> go.Figure:
    """
    Creates a bar chart showing security scores (LOLBAS density) for each topic.
    
    Args:
        topic_scores: DataFrame with security scores per topic
        df_with_topics: DataFrame with topic assignments
        lda_model: Trained LDA model
        
    Returns:
        Plotly Figure object
    """
    from graphs import generate_smart_topic_summary
    
    # Get topic names
    topic_names = []
    lolbas_density = []
    doc_counts = []
    
    for topic_id in topic_scores.index:
        topic_name = generate_smart_topic_summary(topic_id, lda_model, df_with_topics)
        topic_names.append(f"T{topic_id}: {topic_name}")
        lolbas_density.append(topic_scores.loc[topic_id, 'lolbas_density'])
        doc_counts.append(topic_scores.loc[topic_id, 'total_count'])
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=topic_names,
            y=lolbas_density,
            marker=dict(
                color=lolbas_density,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="LOLBAS<br>Density")
            ),
            text=[f"{d:.1f}%<br>({int(c)} docs)" for d, c in zip(lolbas_density, doc_counts)],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>LOLBAS Density: %{y:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Topic Security Risk Score (LOLBAS Density)',
        xaxis_title='Topics',
        yaxis_title='LOLBAS Density (%)',
        height=500,
        showlegend=False,
        xaxis={'tickangle': -45}
    )
    
    return fig


def create_topic_distribution_chart(df_with_topics: pd.DataFrame, lda_model: LdaMulticore) -> go.Figure:
    """
    Creates an interactive sunburst chart showing topic distribution.
    
    Args:
        df_with_topics: DataFrame with topic assignments
        lda_model: Trained LDA model
        
    Returns:
        Plotly Figure object
    """
    from graphs import generate_smart_topic_summary
    
    # Count documents per topic
    topic_counts = df_with_topics['topic'].value_counts().sort_index()
    
    # Prepare data
    labels = []
    parents = []
    values = []
    
    # Root
    labels.append("All Commands")
    parents.append("")
    values.append(len(df_with_topics))
    
    # Topics
    for topic_id, count in topic_counts.items():
        topic_name = generate_smart_topic_summary(topic_id, lda_model, df_with_topics)
        labels.append(f"Topic {topic_id}: {topic_name}")
        parents.append("All Commands")
        values.append(count)
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentRoot:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Topic Distribution Sunburst',
        height=600
    )
    
    return fig


def create_command_length_distribution(df_with_topics: pd.DataFrame) -> go.Figure:
    """
    Creates a box plot showing command length distribution by topic.
    
    Args:
        df_with_topics: DataFrame with topic assignments and tokens
        
    Returns:
        Plotly Figure object
    """
    # Calculate command lengths
    df_plot = df_with_topics.copy()
    df_plot['token_count'] = df_plot['tokens'].apply(len)
    
    fig = go.Figure()
    
    for topic_id in sorted(df_plot['topic'].unique()):
        topic_data = df_plot[df_plot['topic'] == topic_id]['token_count']
        fig.add_trace(go.Box(
            y=topic_data,
            name=f'Topic {topic_id}',
            boxmean='sd'
        ))
    
    fig.update_layout(
        title='Command Length Distribution by Topic',
        xaxis_title='Topic',
        yaxis_title='Number of Tokens',
        height=500,
        showlegend=True
    )
    
    return fig


def create_comprehensive_index(
    df_with_topics,
    lda_model,
    topic_scores,
    lolbas_keywords,
    output_dir,
    total_processing_time=None
):
    """
    Creates a comprehensive index.html that serves as the main entry point to all analysis results.
    This is the primary deliverable showcasing the full analytical journey.
    
    Args:
        df_with_topics: DataFrame with all analysis results
        lda_model: Trained LDA model
        topic_scores: Security analysis scores
        lolbas_keywords: List of LOLBAS keywords
        output_dir: Directory to save all output files
        total_processing_time: Total time taken for analysis (optional)
    """
    import os
    from pathlib import Path
    import json
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    treemap_fig = create_topic_treemap_gensim(df_with_topics, lda_model)
    heatmap_fig = create_topic_heatmap(df_with_topics, lda_model)
    sunburst_fig = create_topic_distribution_chart(df_with_topics, lda_model)
    length_fig = create_command_length_distribution(df_with_topics)
    
    security_fig = None
    if topic_scores is not None:
        security_fig = create_security_score_chart(topic_scores, df_with_topics, lda_model)
    
    # Generate summary statistics
    total_commands = len(df_with_topics)
    unique_topics = df_with_topics['topic'].nunique()
    
    # Calculate key metrics
    lolbas_count = df_with_topics['contains_lolbas'].sum() if 'contains_lolbas' in df_with_topics.columns else 0
    lolbas_percentage = (lolbas_count / total_commands * 100) if total_commands > 0 else 0
    
    mitre_count = 0
    if 'mitre_count' in df_with_topics.columns:
        mitre_count = (df_with_topics['mitre_count'] > 0).sum()
    
    avg_complexity = df_with_topics['complexity_score'].mean() if 'complexity_score' in df_with_topics.columns else 0
    
    # Get top 3 riskiest topics
    high_risk_topics = []
    if topic_scores is not None:
        for topic_id, row in topic_scores.head(3).iterrows():
            topic_name = generate_smart_topic_summary(topic_id, lda_model, df_with_topics)
            high_risk_topics.append({
                'id': topic_id,
                'name': topic_name,
                'risk_score': row['risk_score'],
                'lolbas_count': int(row['lolbas_count']),
                'doc_count': int(row['total_count'])
            })
    
    # Create comprehensive index.html
    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TCLaaC Analysis Report - Command Line Security Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #2d3748;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .hero {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
        }}
        
        .hero h1 {{
            font-size: 3em;
            margin-bottom: 15px;
            font-weight: 800;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        
        .hero .subtitle {{
            font-size: 1.4em;
            opacity: 0.95;
            margin-bottom: 10px;
        }}
        
        .hero .timestamp {{
            font-size: 1em;
            opacity: 0.8;
            margin-top: 20px;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            padding: 40px;
            background: #f7fafc;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        .card .icon {{
            font-size: 3em;
            margin-bottom: 15px;
        }}
        
        .card .value {{
            font-size: 2.5em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .card .label {{
            font-size: 1.1em;
            color: #718096;
            font-weight: 500;
        }}
        
        .card .detail {{
            font-size: 0.9em;
            color: #a0aec0;
            margin-top: 8px;
        }}
        
        .section {{
            padding: 40px;
        }}
        
        .section-title {{
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 25px;
            color: #2d3748;
            border-left: 6px solid #667eea;
            padding-left: 20px;
        }}
        
        .risk-topics {{
            display: grid;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .risk-topic {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            border-left: 5px solid #f56565;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        
        .risk-topic.high {{
            border-left-color: #f56565;
            background: linear-gradient(90deg, #fff5f5 0%, white 100%);
        }}
        
        .risk-topic.medium {{
            border-left-color: #ed8936;
            background: linear-gradient(90deg, #fffaf0 0%, white 100%);
        }}
        
        .risk-topic.low {{
            border-left-color: #48bb78;
            background: linear-gradient(90deg, #f0fff4 0%, white 100%);
        }}
        
        .risk-topic h3 {{
            font-size: 1.4em;
            margin-bottom: 10px;
            color: #2d3748;
        }}
        
        .risk-topic .metrics {{
            display: flex;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .risk-topic .metric {{
            display: flex;
            flex-direction: column;
        }}
        
        .risk-topic .metric-value {{
            font-size: 1.5em;
            font-weight: 700;
            color: #667eea;
        }}
        
        .risk-topic .metric-label {{
            font-size: 0.9em;
            color: #718096;
        }}
        
        .action-buttons {{
            display: flex;
            gap: 20px;
            margin-top: 30px;
            flex-wrap: wrap;
        }}
        
        .btn {{
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }}
        
        .btn-secondary {{
            background: #4a5568;
            color: white;
        }}
        
        .btn-secondary:hover {{
            background: #2d3748;
            transform: translateY(-2px);
        }}
        
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }}
        
        .feature-card {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            text-align: center;
            transition: all 0.3s;
        }}
        
        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        .feature-card .icon {{
            font-size: 3.5em;
            margin-bottom: 20px;
        }}
        
        .feature-card h3 {{
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #2d3748;
        }}
        
        .feature-card p {{
            color: #718096;
            line-height: 1.6;
        }}
        
        .footer {{
            background: #2d3748;
            color: white;
            padding: 30px 40px;
            text-align: center;
        }}
        
        .footer a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }}
        
        .footer a:hover {{
            text-decoration: underline;
        }}
        
        .methodology {{
            background: #edf2f7;
            padding: 30px;
            border-radius: 10px;
            margin-top: 30px;
        }}
        
        .methodology h3 {{
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #2d3748;
        }}
        
        .methodology ol {{
            margin-left: 20px;
            line-height: 2;
            color: #4a5568;
        }}
        
        .badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-right: 8px;
        }}
        
        .badge-danger {{
            background: #fed7d7;
            color: #c53030;
        }}
        
        .badge-warning {{
            background: #feebc8;
            color: #c05621;
        }}
        
        .badge-success {{
            background: #c6f6d5;
            color: #276749;
        }}
        
        .badge-info {{
            background: #bee3f8;
            color: #2c5282;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Hero Section -->
        <div class="hero">
            <h1>🔍 TCLaaC Analysis Report</h1>
            <p class="subtitle">The Command Line as a Corpus - Comprehensive Security Analysis</p>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {f'<p class="timestamp">Processing Time: {total_processing_time:.2f}s ({total_processing_time/60:.2f} minutes)</p>' if total_processing_time else ''}
        </div>
        
        <!-- Summary Cards -->
        <div class="summary-cards">
            <div class="card">
                <div class="icon">📊</div>
                <div class="value">{total_commands:,}</div>
                <div class="label">Total Commands Analyzed</div>
            </div>
            
            <div class="card">
                <div class="icon">🏷️</div>
                <div class="value">{unique_topics}</div>
                <div class="label">Discovered Topics</div>
                <div class="detail">Distinct command patterns</div>
            </div>
            
            <div class="card">
                <div class="icon">🚫</div>
                <div class="value">{lolbas_count:,}</div>
                <div class="label">LOLBAS Detections</div>
                <div class="detail">{lolbas_percentage:.1f}% of commands</div>
            </div>
            
            <div class="card">
                <div class="icon">🎯</div>
                <div class="value">{mitre_count:,}</div>
                <div class="label">MITRE ATT&CK Matches</div>
                <div class="detail">Technique mappings found</div>
            </div>
            
            <div class="card">
                <div class="icon">⚡</div>
                <div class="value">{avg_complexity:.0f}/100</div>
                <div class="label">Avg Complexity Score</div>
                <div class="detail">Command obfuscation level</div>
            </div>
        </div>
        
        <!-- High Risk Topics Section -->
        <div class="section">
            <h2 class="section-title">🔴 High Risk Topics</h2>
            <p style="color: #718096; margin-bottom: 20px;">Topics with elevated security risk based on LOLBAS density, MITRE ATT&CK coverage, and command complexity.</p>
            
            <div class="risk-topics">"""
    
    # Add high risk topic cards
    for i, topic in enumerate(high_risk_topics):
        risk_class = 'high' if i == 0 else ('medium' if i == 1 else 'low')
        risk_badge = f'<span class="badge badge-danger">Critical</span>' if i == 0 else (f'<span class="badge badge-warning">High</span>' if i == 1 else f'<span class="badge badge-info">Medium</span>')
        
        index_html += f"""
                <div class="risk-topic {risk_class}">
                    <h3>Topic #{topic['id']}: {topic['name']} {risk_badge}</h3>
                    <div class="metrics">
                        <div class="metric">
                            <span class="metric-value">{topic['risk_score']:.1f}</span>
                            <span class="metric-label">Risk Score</span>
                        </div>
                        <div class="metric">
                            <span class="metric-value">{topic['lolbas_count']}</span>
                            <span class="metric-label">LOLBAS Commands</span>
                        </div>
                        <div class="metric">
                            <span class="metric-value">{topic['doc_count']}</span>
                            <span class="metric-label">Total Commands</span>
                        </div>
                    </div>
                </div>"""
    
    index_html += """
            </div>
        </div>
        
        <!-- Analysis Features Section -->
        <div class="section" style="background: #f7fafc;">
            <h2 class="section-title">🎨 Interactive Visualizations</h2>
            <p style="color: #718096; margin-bottom: 30px;">Explore the analysis results through multiple interactive visualizations that reveal command patterns, security risks, and topic distributions.</p>
            
            <div class="action-buttons">
                <a href="analysis_dashboard.html" class="btn btn-primary">
                    📊 Open Interactive Dashboard
                </a>
                <a href="topic_summary.csv" class="btn btn-secondary">
                    📥 Download Topic Summary (CSV)
                </a>
                <a href="analysis_dataframe.parquet" class="btn btn-secondary">
                    💾 Download Full Results (Parquet)
                </a>
            </div>
            
            <div class="features-grid">
                <div class="feature-card">
                    <div class="icon">🌳</div>
                    <h3>Topic Treemap</h3>
                    <p>Hierarchical view of command groups organized by topic with fuzzy matching for similar commands.</p>
                </div>
                
                <div class="feature-card">
                    <div class="icon">🔥</div>
                    <h3>Word Heatmap</h3>
                    <p>Distribution of top words across all topics revealing key patterns and terminology.</p>
                </div>
                
                <div class="feature-card">
                    <div class="icon">🛡️</div>
                    <h3>Security Risk Analysis</h3>
                    <p>Comprehensive risk scoring based on LOLBAS density, MITRE ATT&CK techniques, and complexity.</p>
                </div>
                
                <div class="feature-card">
                    <div class="icon">🌟</div>
                    <h3>Topic Distribution</h3>
                    <p>Sunburst visualization showing the proportion of commands in each discovered topic.</p>
                </div>
                
                <div class="feature-card">
                    <div class="icon">📏</div>
                    <h3>Complexity Analysis</h3>
                    <p>Command length and complexity distribution by topic to identify obfuscation patterns.</p>
                </div>
                
                <div class="feature-card">
                    <div class="icon">🎯</div>
                    <h3>LOLBAS Filtering</h3>
                    <p>Dynamic filtering to focus on or exclude Living-off-the-Land binary usage.</p>
                </div>
            </div>
        </div>
        
        <!-- Methodology Section -->
        <div class="section">
            <h2 class="section-title">🔬 Analysis Methodology</h2>
            
            <div class="methodology">
                <h3>Pipeline Workflow</h3>
                <ol>
                    <li><strong>Data Loading & Enrichment</strong> - Load Sysmon Event ID 1 logs and enrich with LOLBAS command examples</li>
                    <li><strong>Text Preprocessing</strong> - Normalize commands (GUIDs, IPs, paths) and tokenize using custom regex patterns</li>
                    <li><strong>Topic Modeling</strong> - Train LDA model with hyperparameter tuning to discover latent command patterns</li>
                    <li><strong>Security Analysis</strong> - Map commands to MITRE ATT&CK techniques and calculate risk scores</li>
                    <li><strong>Visualization Generation</strong> - Create interactive dashboards showcasing all analysis results</li>
                </ol>
                
                <h3 style="margin-top: 25px;">Risk Scoring Formula</h3>
                <p style="color: #4a5568; margin-top: 10px;">
                    Risk Score = (LOLBAS Density × 0.4) + (MITRE Coverage × 10 × 0.3) + (Avg Complexity × 0.15) + (Unique Binaries × 2 × 0.15)
                </p>
            </div>
        </div>
        
        <!-- Key Insights Section -->
        <div class="section" style="background: #f7fafc;">
            <h2 class="section-title">💡 Key Insights</h2>
            
            <div class="features-grid">
                <div class="feature-card">
                    <div class="icon">📈</div>
                    <h3>Pattern Discovery</h3>
                    <p>Identified {unique_topics} distinct command-line usage patterns through unsupervised learning.</p>
                </div>
                
                <div class="feature-card">
                    <div class="icon">🎯</div>
                    <h3>LOLBAS Coverage</h3>
                    <p>{lolbas_percentage:.1f}% of analyzed commands utilize Living-off-the-Land binaries.</p>
                </div>
                
                <div class="feature-card">
                    <div class="icon">🔍</div>
                    <h3>Attack Techniques</h3>
                    <p>Mapped {mitre_count:,} commands to known MITRE ATT&CK techniques for threat intelligence.</p>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p style="font-size: 1.1em; margin-bottom: 10px;">
                <strong>TCLaaC - The Command Line as a Corpus</strong>
            </p>
            <p style="margin-bottom: 15px;">
                NLP-based Security Log Analysis using LDA Topic Modeling
            </p>
            <p>
                <a href="https://github.com/CampbellTrevor/TCLaaC" target="_blank">GitHub Repository</a> | 
                <a href="https://lolbas-project.github.io/" target="_blank">LOLBAS Project</a> | 
                <a href="https://attack.mitre.org/" target="_blank">MITRE ATT&CK</a>
            </p>
            <p style="margin-top: 20px; opacity: 0.7; font-size: 0.9em;">
                © 2025 | Built with Python, Gensim, Plotly, and scikit-learn
            </p>
        </div>
    </div>
</body>
</html>"""
    
    # Write index.html
    index_path = output_path / 'index.html'
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    # Also create the analysis_dashboard.html (SPA with all plots)
    create_analysis_spa(
        treemap_fig=treemap_fig,
        heatmap_fig=heatmap_fig,
        security_fig=security_fig,
        sunburst_fig=sunburst_fig,
        length_fig=length_fig,
        lolbas_keywords=lolbas_keywords,
        output_path=str(output_path / 'analysis_dashboard.html')
    )
    
    return str(index_path)


def create_analysis_spa(treemap_fig, heatmap_fig, security_fig, sunburst_fig, length_fig, lolbas_keywords, output_path):
    """
    Creates a Single Page Application (SPA) combining all visualizations with tabs.
    Includes dynamic LOLBAS filtering for the treemap.
    
    Args:
        treemap_fig: Plotly treemap figure
        heatmap_fig: Plotly heatmap figure
        security_fig: Plotly security chart figure (or None)
        sunburst_fig: Plotly sunburst figure
        length_fig: Plotly box plot figure
        lolbas_keywords: List of LOLBAS keywords for filtering
        output_path: Path to save the HTML file
    """
    import json
    
    # Convert figures to JSON for embedding
    treemap_json = treemap_fig.to_json()
    heatmap_json = heatmap_fig.to_json()
    security_json = security_fig.to_json() if security_fig else 'null'
    sunburst_json = sunburst_fig.to_json()
    length_json = length_fig.to_json()
    
    # Escape LOLBAS keywords for JavaScript
    lolbas_json = json.dumps(lolbas_keywords)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TCLaaC Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
            overflow-x: auto;
        }}
        
        .tab {{
            padding: 18px 30px;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 16px;
            font-weight: 500;
            color: #6c757d;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
            white-space: nowrap;
        }}
        
        .tab:hover {{
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
        }}
        
        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            background: white;
        }}
        
        .tab-content {{
            display: none;
            padding: 30px;
            animation: fadeIn 0.3s;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .plot-container {{
            width: 100%;
            min-height: 600px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            margin-bottom: 20px;
        }}
        
        .controls {{
            margin-bottom: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }}
        
        .controls h3 {{
            margin-bottom: 15px;
            color: #495057;
            font-size: 1.1em;
        }}
        
        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-right: 10px;
            margin-bottom: 10px;
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .btn-secondary {{
            background: #6c757d;
            color: white;
        }}
        
        .btn-secondary:hover {{
            background: #5a6268;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(108, 117, 125, 0.4);
        }}
        
        .info-box {{
            padding: 15px;
            background: #e7f3ff;
            border-left: 4px solid #667eea;
            border-radius: 4px;
            margin-bottom: 20px;
            color: #004085;
        }}
        
        .footer {{
            padding: 20px 30px;
            background: #f8f9fa;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
            border-top: 1px solid #e9ecef;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
            margin-left: 8px;
        }}
        
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 TCLaaC Analysis Dashboard</h1>
            <p>The Command Line as a Corpus - Interactive Topic Analysis</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('treemap')">
                📊 Topic Treemap
            </button>
            <button class="tab" onclick="showTab('heatmap')">
                🔥 Word Heatmap
            </button>
            <button class="tab" onclick="showTab('security')">
                🛡️ Security Risk
            </button>
            <button class="tab" onclick="showTab('sunburst')">
                🌟 Distribution
            </button>
            <button class="tab" onclick="showTab('length')">
                📏 Complexity
            </button>
        </div>
        
        <div id="treemap" class="tab-content active">
            <div class="info-box">
                <strong>ℹ️ Topic Treemap:</strong> Hierarchical visualization of command groups by topic. 
                Use the controls below to filter LOLBAS commands dynamically.
            </div>
            <div class="controls">
                <h3>Filter Controls</h3>
                <button class="btn btn-primary" onclick="filterLOLBAS()">
                    🚫 Hide LOLBAS Commands
                </button>
                <button class="btn btn-secondary" onclick="showAll()">
                    👁️ Show All Commands
                </button>
                <span id="filter-status" class="badge badge-success">Showing all commands</span>
            </div>
            <div id="treemap-plot" class="plot-container"></div>
        </div>
        
        <div id="heatmap" class="tab-content">
            <div class="info-box">
                <strong>ℹ️ Topic-Word Heatmap:</strong> Distribution of top words across topics. 
                Darker colors indicate higher word probability in that topic.
            </div>
            <div id="heatmap-plot" class="plot-container"></div>
        </div>
        
        <div id="security" class="tab-content">
            <div class="info-box">
                <strong>ℹ️ Security Risk Analysis:</strong> LOLBAS density scores for each topic. 
                Higher scores indicate more suspicious activity patterns.
            </div>
            <div id="security-plot" class="plot-container"></div>
        </div>
        
        <div id="sunburst" class="tab-content">
            <div class="info-box">
                <strong>ℹ️ Topic Distribution:</strong> Hierarchical view showing the proportion of commands in each topic.
                Click segments to zoom in.
            </div>
            <div id="sunburst-plot" class="plot-container"></div>
        </div>
        
        <div id="length" class="tab-content">
            <div class="info-box">
                <strong>ℹ️ Command Complexity:</strong> Distribution of command length (token count) by topic.
                Box plots show median, quartiles, and outliers.
            </div>
            <div id="length-plot" class="plot-container"></div>
        </div>
        
        <div class="footer">
            Generated by TCLaaC (The Command Line as a Corpus) | 
            © 2025 | 
            <a href="https://github.com/CampbellTrevor/TCLaaC" target="_blank" style="color: #667eea;">GitHub</a>
        </div>
    </div>
    
    <script>
        // Store the original treemap data
        let treemapData = {treemap_json};
        let originalTreemapData = JSON.parse(JSON.stringify(treemapData));
        let lolbasKeywords = {lolbas_json};
        let isFiltered = false;
        
        // Initialize all plots
        function initPlots() {{
            // Treemap
            Plotly.newPlot('treemap-plot', treemapData.data, treemapData.layout, {{responsive: true}});
            
            // Heatmap
            let heatmapData = {heatmap_json};
            Plotly.newPlot('heatmap-plot', heatmapData.data, heatmapData.layout, {{responsive: true}});
            
            // Security (if available)
            let securityData = {security_json};
            if (securityData !== null) {{
                Plotly.newPlot('security-plot', securityData.data, securityData.layout, {{responsive: true}});
            }} else {{
                document.getElementById('security-plot').innerHTML = '<div style="padding: 40px; text-align: center; color: #6c757d;"><h3>Security analysis not available</h3><p>LOLBAS repository not found during analysis.</p></div>';
            }}
            
            // Sunburst
            let sunburstData = {sunburst_json};
            Plotly.newPlot('sunburst-plot', sunburstData.data, sunburstData.layout, {{responsive: true}});
            
            // Length
            let lengthData = {length_json};
            Plotly.newPlot('length-plot', lengthData.data, lengthData.layout, {{responsive: true}});
        }}
        
        // Tab switching
        function showTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
        
        // Filter LOLBAS commands from treemap
        function filterLOLBAS() {{
            if (isFiltered) return;
            
            // Create a case-insensitive regex pattern from LOLBAS keywords
            let pattern = new RegExp(lolbasKeywords.map(k => k.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&')).join('|'), 'i');
            
            // Filter the treemap data
            let filteredData = JSON.parse(JSON.stringify(originalTreemapData));
            
            // Filter labels and values
            if (filteredData.data && filteredData.data[0]) {{
                let labels = filteredData.data[0].labels || [];
                let parents = filteredData.data[0].parents || [];
                let values = filteredData.data[0].values || [];
                
                let newLabels = [];
                let newParents = [];
                let newValues = [];
                
                for (let i = 0; i < labels.length; i++) {{
                    let label = labels[i];
                    // Keep if it doesn't match LOLBAS pattern or is a parent node
                    if (!pattern.test(label) || parents[i] === "" || label.includes("Topic")) {{
                        newLabels.push(label);
                        newParents.push(parents[i]);
                        newValues.push(values[i]);
                    }}
                }}
                
                filteredData.data[0].labels = newLabels;
                filteredData.data[0].parents = newParents;
                filteredData.data[0].values = newValues;
            }}
            
            Plotly.newPlot('treemap-plot', filteredData.data, filteredData.layout, {{responsive: true}});
            
            document.getElementById('filter-status').textContent = 'LOLBAS commands hidden';
            document.getElementById('filter-status').className = 'badge badge-warning';
            isFiltered = true;
        }}
        
        // Show all commands
        function showAll() {{
            Plotly.newPlot('treemap-plot', originalTreemapData.data, originalTreemapData.layout, {{responsive: true}});
            
            document.getElementById('filter-status').textContent = 'Showing all commands';
            document.getElementById('filter-status').className = 'badge badge-success';
            isFiltered = false;
        }}
        
        // Initialize on load
        window.onload = initPlots;
    </script>
</body>
</html>"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)