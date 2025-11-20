"""
Network graph visualization for command-line relationships.

This module provides network graph visualizations showing:
- Command co-occurrence patterns
- Topic relationships
- MITRE technique connections
- Execution flow analysis
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import plotly.graph_objects as go
import networkx as nx

logger = logging.getLogger(__name__)


def create_command_network(
    df_with_topics: pd.DataFrame,
    min_edge_weight: int = 2,
    max_nodes: int = 50
) -> go.Figure:
    """
    Create a network graph showing command co-occurrence patterns.
    
    Commands that appear together in the same topic are connected.
    Edge weight represents co-occurrence frequency.
    
    Args:
        df_with_topics: DataFrame with topic assignments
        min_edge_weight: Minimum co-occurrence count to include edge
        max_nodes: Maximum number of nodes to include
        
    Returns:
        Plotly Figure with network graph
    """
    logger.info("Creating command co-occurrence network...")
    
    # Build co-occurrence matrix
    topic_groups = df_with_topics.groupby('topic')['root_command'].apply(list)
    
    # Count co-occurrences
    co_occurrence = defaultdict(int)
    command_counts = Counter()
    
    for commands in topic_groups:
        unique_commands = list(set(commands))
        command_counts.update(unique_commands)
        
        # Count pairs
        for i, cmd1 in enumerate(unique_commands):
            for cmd2 in unique_commands[i+1:]:
                pair = tuple(sorted([cmd1, cmd2]))
                co_occurrence[pair] += 1
    
    # Filter top commands
    top_commands = [cmd for cmd, _ in command_counts.most_common(max_nodes)]
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes
    for cmd in top_commands:
        G.add_node(cmd, weight=command_counts[cmd])
    
    # Add edges
    for (cmd1, cmd2), weight in co_occurrence.items():
        if weight >= min_edge_weight and cmd1 in top_commands and cmd2 in top_commands:
            G.add_edge(cmd1, cmd2, weight=weight)
    
    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    
    if len(G.nodes()) == 0:
        logger.warning("  No connections found, returning empty graph")
        return go.Figure()
    
    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Extract node and edge positions
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight*0.5, color='rgba(125, 125, 125, 0.5)'),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Count: {G.nodes[node]['weight']}")
        node_size.append(min(50, G.nodes[node]['weight'] * 2))
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition='top center',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        textfont=dict(size=10)
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace])
    
    fig.update_layout(
        title=dict(text="Command Co-occurrence Network", font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    logger.info(f"  Network graph created: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    return fig


def create_topic_relationship_network(
    df_with_topics: pd.DataFrame,
    lda_model
) -> go.Figure:
    """
    Create a network showing relationships between topics.
    
    Topics are connected based on command similarity and shared features.
    
    Args:
        df_with_topics: DataFrame with topic assignments
        lda_model: Trained LDA model
        
    Returns:
        Plotly Figure with topic network
    """
    logger.info("Creating topic relationship network...")
    
    num_topics = lda_model.num_topics
    
    # Calculate topic similarity matrix using Hellinger distance
    topic_similarities = np.zeros((num_topics, num_topics))
    
    for i in range(num_topics):
        for j in range(i+1, num_topics):
            # Get topic word distributions
            topic_i = dict(lda_model.show_topic(i, topn=100))
            topic_j = dict(lda_model.show_topic(j, topn=100))
            
            # Calculate overlap (Jaccard similarity of top words)
            words_i = set(topic_i.keys())
            words_j = set(topic_j.keys())
            
            if len(words_i | words_j) > 0:
                similarity = len(words_i & words_j) / len(words_i | words_j)
                topic_similarities[i, j] = similarity
                topic_similarities[j, i] = similarity
    
    # Create network
    G = nx.Graph()
    
    # Add nodes (topics)
    for i in range(num_topics):
        topic_size = (df_with_topics['topic'] == i).sum()
        G.add_node(i, size=topic_size)
    
    # Add edges (topic similarities)
    threshold = 0.1  # Minimum similarity to create edge
    for i in range(num_topics):
        for j in range(i+1, num_topics):
            if topic_similarities[i, j] > threshold:
                G.add_edge(i, j, weight=topic_similarities[i, j])
    
    # Layout
    pos = nx.spring_layout(G, k=1.0, iterations=50)
    
    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight*10, color='rgba(125, 125, 125, 0.3)'),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get top words for topic
        top_words = [word for word, _ in lda_model.show_topic(node, topn=5)]
        node_text.append(
            f"Topic {node}<br>"
            f"Size: {G.nodes[node]['size']} docs<br>"
            f"Keywords: {', '.join(top_words)}"
        )
        node_size.append(min(80, G.nodes[node]['size'] * 2))
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[f"T{node}" for node in G.nodes()],
        textposition='middle center',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=[G.nodes[node]['size'] for node in G.nodes()],
            colorscale='Viridis',
            line=dict(width=2, color='white'),
            colorbar=dict(title="Doc Count")
        ),
        textfont=dict(size=12, color='white')
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace])
    
    fig.update_layout(
        title=dict(text="Topic Relationship Network", font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    logger.info(f"  Topic network created: {len(G.nodes())} topics, {len(G.edges())} connections")
    
    return fig


def create_mitre_attack_network(
    df_with_topics: pd.DataFrame
) -> go.Figure:
    """
    Create a network showing MITRE ATT&CK technique relationships.
    
    Techniques are connected if they appear together in commands.
    
    Args:
        df_with_topics: DataFrame with MITRE technique mappings
        
    Returns:
        Plotly Figure with MITRE network
    """
    logger.info("Creating MITRE ATT&CK technique network...")
    
    if 'mitre_techniques' not in df_with_topics.columns:
        logger.warning("  No MITRE techniques found, skipping")
        return go.Figure()
    
    # Build co-occurrence matrix for techniques
    co_occurrence = defaultdict(int)
    technique_counts = Counter()
    
    for techniques in df_with_topics['mitre_techniques']:
        if isinstance(techniques, str) and techniques:
            techs = techniques.split(',')
            technique_counts.update(techs)
            
            # Count pairs
            for i, t1 in enumerate(techs):
                for t2 in techs[i+1:]:
                    pair = tuple(sorted([t1.strip(), t2.strip()]))
                    co_occurrence[pair] += 1
    
    if len(technique_counts) == 0:
        logger.warning("  No technique co-occurrences found")
        return go.Figure()
    
    # Create network
    G = nx.Graph()
    
    # Add nodes (techniques)
    for tech, count in technique_counts.most_common(20):
        G.add_node(tech, count=count)
    
    # Add edges
    for (t1, t2), weight in co_occurrence.items():
        if t1 in G.nodes() and t2 in G.nodes() and weight >= 2:
            G.add_edge(t1, t2, weight=weight)
    
    if len(G.nodes()) == 0:
        return go.Figure()
    
    # Layout
    pos = nx.spring_layout(G, k=0.8, iterations=50)
    
    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight*0.5, color='rgba(255, 0, 0, 0.3)'),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Occurrences: {G.nodes[node]['count']}")
        node_size.append(min(60, G.nodes[node]['count'] * 5))
        node_color.append(G.nodes[node]['count'])
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition='top center',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='Reds',
            line=dict(width=2, color='darkred'),
            colorbar=dict(title="Count")
        ),
        textfont=dict(size=10)
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace])
    
    fig.update_layout(
        title=dict(text="MITRE ATT&CK Technique Co-occurrence Network", font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    logger.info(f"  MITRE network created: {len(G.nodes())} techniques, {len(G.edges())} connections")
    
    return fig
