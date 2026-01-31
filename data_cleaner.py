"""
Data Cleaning and Preprocessing Utility

This module provides robust data cleaning functions for pandas DataFrames.
"""

import pandas as pd
import numpy as np
import re
from typing import Any


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically clean and preprocess a DataFrame.
    
    This function performs the following operations:
    1. Standardizes column headers to snake_case
    2. Detects and converts date/time columns to datetime objects
    3. Handles missing values (0 for numeric, 'Unknown' for strings)
    4. Removes duplicate rows
    5. Optimizes memory by downcasting numeric columns
    
    Args:
        df (pd.DataFrame): The input DataFrame to clean
        
    Returns:
        pd.DataFrame: The cleaned and optimized DataFrame
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # 1. Standardize Headers to snake_case
    df_clean.columns = [_to_snake_case(col) for col in df_clean.columns]
    
    # 2. Fix Date/Time Columns
    df_clean = _convert_date_columns(df_clean)
    
    # 3. Handle Missing Values
    df_clean = _handle_nulls(df_clean)
    
    # 4. Remove Duplicates
    original_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = original_rows - len(df_clean)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # 5. Optimize Memory Usage
    df_clean = _optimize_dtypes(df_clean)
    
    return df_clean


def _to_snake_case(column_name: str) -> str:
    """
    Convert a column name to snake_case.
    
    Args:
        column_name (str): Original column name
        
    Returns:
        str: Column name in snake_case format
    """
    # Remove special characters except spaces and underscores
    cleaned = re.sub(r'[^\w\s]', '', str(column_name))
    
    # Replace spaces with underscores
    cleaned = re.sub(r'\s+', '_', cleaned)
    
    # Convert camelCase to snake_case
    cleaned = re.sub(r'(?<!^)(?=[A-Z])', '_', cleaned)
    
    # Convert to lowercase and remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned.lower())
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    return cleaned


def _convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and convert date/time columns to datetime objects.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with converted date columns
    """
    date_keywords = ['date', 'time', 'datetime', 'timestamp', 'dt']
    
    for col in df.columns:
        # Check if column name contains date/time keywords
        if any(keyword in col.lower() for keyword in date_keywords):
            try:
                # Try to convert to datetime, handling mixed formats
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                print(f"Converted column '{col}' to datetime")
            except Exception as e:
                print(f"Could not convert column '{col}' to datetime: {e}")
    
    return df


def _handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values based on column data type.
    
    - Numeric columns: Fill with 0
    - String/Object columns: Fill with 'Unknown'
    - Datetime columns: Leave as NaT (not filled)
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with handled nulls
    """
    for col in df.columns:
        if df[col].isnull().any():
            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                null_count = df[col].isnull().sum()
                df[col] = df[col].fillna(0)
                print(f"Filled {null_count} null values in numeric column '{col}' with 0")
            
            # String/Object columns (excluding datetime)
            elif pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
                null_count = df[col].isnull().sum()
                df[col] = df[col].fillna('Unknown')
                print(f"Filled {null_count} null values in string column '{col}' with 'Unknown'")
    
    return df


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Memory-optimized DataFrame
    """
    memory_before = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Downcast integers
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Downcast floats
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    memory_after = df.memory_usage(deep=True).sum() / 1024**2  # MB
    memory_saved = memory_before - memory_after
    
    if memory_saved > 0:
        print(f"Memory optimized: {memory_before:.2f} MB → {memory_after:.2f} MB (saved {memory_saved:.2f} MB)")
    
    return df


# Example usage
if __name__ == "__main__":
    # Create a sample messy DataFrame for testing
    sample_data = {
        'User Name': ['John Doe', 'Jane Smith', 'John Doe', None, 'Bob Wilson'],
        'AGE': [25, None, 25, 30, 45],
        'SignUpDate': ['2023-01-15', '15/02/2023', '2023-01-15', '2023-03-20', None],
        'Total$Sales': [1000.50, 2500.75, 1000.50, None, 5000.25],
        'Status!@#': ['Active', 'Active', 'Active', None, 'Inactive']
    }
    
    df_messy = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df_messy)
    print("\n" + "="*60 + "\n")
    
    # Clean the DataFrame
    df_cleaned = clean_dataframe(df_messy)
    
    print("\n" + "="*60 + "\n")
    print("Cleaned DataFrame:")
    print(df_cleaned)
    print("\nData Types:")
    print(df_cleaned.dtypes)
