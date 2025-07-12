#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Join Analysis CSV with DuckDB Table

This script joins the combined_product_customer_analysis.csv file with 
the iid_return_reasons table in DuckDB and saves the result as a new CSV.
"""

import pandas as pd
import duckdb
import os

def main():
    # Define parameters
    csv_path = "combined_product_customer_analysis.csv"
    output_path = "sample_joined_analysis_GROSS.csv"
    tname = 'iid_return_reasons'
    db_path = 'iid_return_comment'
    row_id = "row_id"
    
    print("NOTE: This script will only keep rows where sentiment analysis was completed")
    
    # Check if input CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: Input file '{csv_path}' not found.")
        return
    
    print(f"Reading analysis data from {csv_path}...")
    # Read the CSV file
    analysis_df = pd.read_csv(csv_path)
    print(f"Loaded {len(analysis_df)} records from analysis CSV")
    
    # Check if the row_id in analysis_df starts with letter prefixes (like 'A1', 'A2')
    # If so, try to extract just the numeric part
    if len(analysis_df) > 0 and isinstance(analysis_df[row_id].iloc[0], str):
        if analysis_df[row_id].iloc[0][0].isalpha():
            print("Detected alphabetic prefix in row_id column of analysis CSV. Extracting numeric part...")
            # Extract numeric part from row_id
            analysis_df['numeric_row_id'] = analysis_df[row_id].str.extract(r'([0-9]+)', expand=False)
            print(f"First few rows of analysis_df after extracting numeric part:")
            print(analysis_df[['numeric_row_id', row_id]].head())
            
            # Use the numeric part for joining
            row_id_for_join = 'numeric_row_id'
        else:
            row_id_for_join = row_id
    else:
        row_id_for_join = row_id
    
    # Connect to DuckDB
    print(f"Connecting to DuckDB database: {db_path}")
    con = duckdb.connect(db_path)
    
    # Check if the table exists
    table_exists = con.execute(f"""
        SELECT count(*) 
        FROM information_schema.tables 
        WHERE table_name = '{tname}'
    """).fetchone()[0]
    
    if not table_exists:
        print(f"Error: Table '{tname}' not found in the database.")
        con.close()
        return
    
    # Read the base data from DuckDB
    print(f"Reading base data from {tname}...")
    base_df = con.execute(f"SELECT * FROM {tname}").fetchdf()
    print(f"Loaded {len(base_df)} records from DuckDB table")
    
    # Check data types of row_id in both DataFrames
    print(f"Data type of row_id in base_df: {base_df[row_id].dtype}")
    print(f"Data type of row_id in analysis_df: {analysis_df[row_id].dtype}")
    
    # Convert row_id to the same type (string) in both DataFrames to ensure compatibility
    base_df[row_id] = base_df[row_id].astype(str)
    
    # If we're using the extracted numeric part for joining
    if row_id_for_join != row_id:
        analysis_df[row_id_for_join] = analysis_df[row_id_for_join].astype(str)
        print(f"Data type of row_id in base_df: {base_df[row_id].dtype}")
        print(f"Data type of numeric_row_id in analysis_df: {analysis_df[row_id_for_join].dtype}")
    else:
        analysis_df[row_id] = analysis_df[row_id].astype(str)
        print(f"Data type of row_id in base_df: {base_df[row_id].dtype}")
        print(f"Data type of row_id in analysis_df: {analysis_df[row_id].dtype}")
    
    # Join the DataFrames
    print("Joining datasets on row_id...")
    if row_id_for_join != row_id:
        # Join on the numeric part from analysis_df and row_id from base_df
        joined_df = pd.merge(
            base_df, 
            analysis_df,
            left_on=row_id,
            right_on=row_id_for_join,
            how='inner'
        )
    else:
        joined_df = pd.merge(
            base_df, 
            analysis_df,
            on=row_id, 
            how='inner'
        )
    
    # Remove rows where sentiment analysis data is missing
    # Get all columns from analysis_df except row_id
    analysis_columns = [c for c in analysis_df.columns if c != row_id]
    
    if analysis_columns:
        # Keep only rows where at least one analysis column has a non-null value
        print("Filtering to keep only rows with completed sentiment analysis...")
        joined_df = joined_df.dropna(subset=analysis_columns, how='all')
    
    # Count records with analysis
    print(f"Result: {joined_df.shape[0]} total records with completed sentiment analysis")
    
    # Save the result to CSV
    print(f"Saving joined data to {output_path}...")
    joined_df.to_csv(output_path, index=False)
    
    print(f"Join completed successfully. Output saved to: {output_path}")
    
    # Close the connection
    con.close()

if __name__ == "__main__":
    main()
