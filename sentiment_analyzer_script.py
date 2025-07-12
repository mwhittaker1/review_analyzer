#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Analyzer Script

This script replicates the functionality from main.ipynb notebook to perform sentiment
analysis on return comments using OpenAI's GPT-4o model.

Usage:
    python sentiment_analyzer_script.py [--config CONFIG_FILE]
"""

import pandas as pd
import duckdb
import openai
import json
import logging
import re
import os
import time
import argparse
import traceback
from datetime import datetime
from io import StringIO
from typing import Optional, Tuple
from google.cloud import secretmanager
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='sentiment_analysis_run.log',
    filemode='a'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sentiment Analysis for Return Comments')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--clear_data', action='store_true', help='Clear existing data on import')
    parser.add_argument('--file_path', help='Path to input CSV file')
    parser.add_argument('--table_name', help='Name of the table to create')
    parser.add_argument('--db_path', help='Path to DuckDB database')
    parser.add_argument('--row_id', help='Column to use as unique identifier')
    parser.add_argument('--comment_column', help='Column containing return comments')
    parser.add_argument('--sample', action='store_true', help='Run analysis on a sample of records')
    parser.add_argument('--no_sample', action='store_true', help='Do not run analysis on a sample (overrides --sample)')
    parser.add_argument('--sample_size', type=int, help='Size of sample to analyze')
    parser.add_argument('--batch_size', type=int, help='Batch size for API calls')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--export_csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--model', help='OpenAI model to use for analysis (e.g., gpt-4o, gpt-4-turbo)')
    parser.add_argument('--import_data', action='store_true', help='Import data from file to database')
    parser.add_argument('--skip_analyzed', action='store_true', help='Skip records that already have sentiment analysis')
    parser.add_argument('--force_reanalyze', action='store_true', help='Force reanalysis of already analyzed records (overrides --skip_analyzed)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def access_secret(secret_path):
    """
    Establishes connection to GCP secret manager and retrieves secret value.
    Ensure authentication is setup for GCP: in bash: gcloud auth application-default login
    """
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(name=secret_path)
    secret_payload = response.payload.data.decode("UTF-8")
    return secret_payload


def import_data(
        fname: str,
        clear=True,
        db_path='temp_db',
        tname='staging_table',
        ftype=None,
):
    """
    Load CSV, Parquet, or Excel file into a DuckDB table.
    Returns confirmation message.
    clear = Replace existing table if true, else create if not exists.
    Adds a unique row_id column to each record.
    """
    import os
    con = duckdb.connect(db_path)

    # Infer file type if not provided
    if ftype is None:
        ext = os.path.splitext(fname)[1].lower()
        if ext == '.csv':
            ftype = 'csv'
        elif ext == '.parquet':
            ftype = 'parquet'
        elif ext in ('.xlsx', '.xls'):
            ftype = 'excel'
        else:
            con.close()
            raise ValueError("Unsupported file extension.")

    if ftype == 'csv':
        mode = 'OR REPLACE TABLE' if clear else 'TABLE IF NOT EXISTS'
        try:
            # First attempt with standard options, add row_id
            con.execute(f"""
                CREATE {mode} {tname} AS
                SELECT *, ROW_NUMBER() OVER () AS row_id FROM read_csv_auto('{fname}', escape='\\', encoding='utf-8', header=True)
            """)
        except Exception as e:
            logger.warning(f"First attempt failed, trying with strict_mode=false: {str(e)}")
            try:
                # Second attempt with strict_mode=false to handle unterminated quotes
                con.execute(f"""
                    CREATE {mode} {tname} AS
                    SELECT *, ROW_NUMBER() OVER () AS row_id FROM read_csv_auto(
                        '{fname}', 
                        escape='\\',
                        encoding='utf-8',
                        header=True,
                        strict_mode=false
                    )
                """)
            except Exception as e2:
                logger.warning(f"Second attempt failed, trying with ignore_errors=true: {str(e2)}")
                try:
                    # Third attempt with ignore_errors=true to skip problematic rows
                    con.execute(f"""
                        CREATE {mode} {tname} AS
                        SELECT *, ROW_NUMBER() OVER () AS row_id FROM read_csv_auto(
                            '{fname}', 
                            escape='\\',
                            encoding='utf-8',
                            header=True,
                            strict_mode=false,
                            ignore_errors=true
                        )
                    """)
                except Exception as e3:
                    con.close()
                    raise ValueError(f"Failed to import CSV after multiple attempts: {str(e3)}")
    elif ftype == 'parquet':
        mode = 'OR REPLACE TABLE' if clear else 'TABLE IF NOT EXISTS'
        con.execute(f"""
            CREATE {mode} {tname} AS
            SELECT *, ROW_NUMBER() OVER () AS row_id FROM read_parquet('{fname}')
        """)
    elif ftype == 'excel':
        df = pd.read_excel(fname)
        if clear:
            con.execute(f"DROP TABLE IF EXISTS {tname}")
        # Add row_id column to DataFrame
        df['row_id'] = range(1, len(df) + 1)
        con.register('temp_excel_df', df)
        con.execute(f"CREATE TABLE {tname} AS SELECT * FROM temp_excel_df")
        con.unregister('temp_excel_df')
    else:
        con.close()
        raise ValueError("Unsupported file type.")

    con.close()
    logger.info(f"Import completed: {fname} into {tname} at {db_path}")
    return f"Import completed: {fname} into {tname} at {db_path}"


def fetch_return_comments(con, tname, is_sample=False, sample_size=500, comment_column='RETURN_COMMENT', row_id_column='row_id', skip_analyzed=True):
    """
    Extract return comments from the DuckDB table with a row_id for later matching.
    con: DuckDB connection
    tname: Table name
    is_sample: Whether to take a sample or full dataset
    sample_size: Number of records to sample
    comment_column: Column containing return comments
    row_id_column: Column to use for joining results back to main dataset
    skip_analyzed: Skip records that already have sentiment analysis
    
    Returns:
        DataFrame with comments and row_id
    """
    # Filter for non-empty comments
    comment_filter = f"""WHERE "{comment_column}" IS NOT NULL AND TRIM("{comment_column}") != ''"""
    
    # Check if we should skip already analyzed records
    if skip_analyzed:
        # Check if the sentiment table exists
        sentiment_table = f"{tname}_with_sentiment"
        sentiment_exists = con.execute(f"""
            SELECT count(*) 
            FROM information_schema.tables 
            WHERE table_name = '{sentiment_table}'
        """).fetchone()[0]
        
        if sentiment_exists:
            # Get records that don't have sentiment analysis yet
            query = f"""
            SELECT 
                "{row_id_column}" as row_id,
                "{comment_column}" as comment
            FROM {tname} as base
            {comment_filter}
            AND NOT EXISTS (
                SELECT 1 
                FROM {sentiment_table} as sent 
                WHERE sent.row_id = base.{row_id_column}
            )
            """
            if is_sample:
                query += f" ORDER BY RANDOM() LIMIT {sample_size}"
            
            logger.info(f"Fetching only records without existing sentiment analysis")
            result = con.execute(query).df()
            row_count = len(result)
            logger.info(f"Extracted {row_count} comments without existing sentiment analysis from {tname}")
            
            if row_count == 0:
                logger.info("All records in the database have already been analyzed")
            
            return result
    
    # Standard query without checking for existing sentiment
    if is_sample:
        sample_query = f"ORDER BY RANDOM() LIMIT {sample_size}"  
    else:
        sample_query = ""
    
    # Select the comment, row_id, and add row identifier
    query = f"""
    SELECT 
        "{row_id_column}" as row_id,
        "{comment_column}" as comment
    FROM {tname}
    {comment_filter}
    {sample_query}
    """
    
    result = con.execute(query).df()
    logger.info(f"Extracted {len(result)} comments from {tname}")
    
    return result


def prepare_data_for_analysis(text):
    """Sanitize comment text and strip code block formatting from model output."""
    if not isinstance(text, str):
        return ""
    # Remove control characters and excessive whitespace
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def strip_code_block(text):
    """Remove Markdown code block wrappers from OpenAI response."""
    text = text.strip()
    code_block_pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(code_block_pattern, text)
    if match:
        return match.group(1).strip()
    return text


def ai_analyze_comments(client, prompt: str, df: pd.DataFrame, model: str = 'gpt-4o', debug: bool = False) -> str:
    """
    Sends `prompt` plus the JSON version of `df` to ChatGPT,
    and returns the model's response.strip()
    """
    df_json = df.to_json(orient="records")
    if debug:
        logger.debug("Prompt sent to model:\n%s", prompt)

    messages = [
        {"role": "system",
            "content": 
            "You are an expert linguistic analyst specializing in extracting and scoring themes from customer return comments. "
            "You always return your output as a single JSON array of objects, one per input record, using exactly the keys and structure specified in the user's instructions. "
            "Do not include any explanations, extra text, or formatting outside the required JSON array. "
            "Be precise, consistent, and strictly follow the output schema and scoring rules provided."
        },
        {"role": "user", "content": prompt},
        {"role": "user", "content": df_json}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )

    content = resp.choices[0].message.content.strip()

    if debug:
        logger.debug("Raw response from OpenAI:\n%s", content)
    if not content:
        raise ValueError("Empty response from OpenAI")

    return content


def handle_sentiment_analysis(
    comments_df,
    client=None,
    product_prompt_path: str = 'prompts/product_prompt.txt',
    customer_prompt_path: str = 'prompts/customer_sentiment_prompt.txt',
    batch_size: int = 750,
    debug: bool = False,
    export_csv: bool = True,
    id_column: str = 'row_id',
    output_dir: str = 'batches',
    model: str = 'gpt-4o',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run both Product and Customer analysis on same data set, batching results and saving each batch to CSV.
    At the end, create a combined CSV of all results.
    """
    # Check if we have data to process
    if len(comments_df) == 0:
        logger.warning("No comments to analyze - empty DataFrame provided")
        print("No comments to analyze - empty DataFrame provided")
        return pd.DataFrame(), pd.DataFrame()
        
    start_time = time.time()
    logger.info(f"Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total records to process: {len(comments_df)}")
    print(f"Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total records to process: {len(comments_df)}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    try:
        with open(product_prompt_path, 'r', encoding='utf-8') as f:
            product_prompt = f.read()
    except FileNotFoundError:
        logger.warning(f"Product prompt file not found at {product_prompt_path}, using default prompt")
        product_prompt = "Analyze the product-related feedback in these return comments and output a JSON array with these fields for each record: row_id, product_issues, size_issues, quality_issues, is_defective, product_sentiment (positive/negative/neutral)."
    
    try:
        with open(customer_prompt_path, 'r', encoding='utf-8') as f:
            customer_prompt = f.read()
    except FileNotFoundError:
        logger.warning(f"Customer prompt file not found at {customer_prompt_path}, using default prompt")
        customer_prompt = "Analyze the customer experience in these return comments and output a JSON array with these fields for each record: row_id, customer_sentiment, refund_requested, exchange_requested, frustration_level (1-5)."

    # Helper to batch DataFrame
    def batch_df(df, batch_size):
        for i in range(0, len(df), batch_size):
            yield i // batch_size + 1, df.iloc[i:i+batch_size]

    # Clean and sanitize comments before batching
    comments_df = comments_df.copy()
    if 'comment' in comments_df.columns:
        comments_df['comment'] = comments_df['comment'].apply(prepare_data_for_analysis)

    # Calculate total batches for progress reporting
    total_batches = (len(comments_df) + batch_size - 1) // batch_size
    
    # PRODUCT FEEDBACK ANALYSIS
    logger.info("\n===== PRODUCT FEEDBACK ANALYSIS =====")
    print("\n===== PRODUCT FEEDBACK ANALYSIS =====")
    product_batches = []
    
    for batch_num, batch_df_ in batch_df(comments_df, batch_size):
        logger.info(f"Processing product batch {batch_num}/{total_batches} with {len(batch_df_)} records...")
        print(f"Processing product batch {batch_num}/{total_batches} with {len(batch_df_)} records... ({batch_num/total_batches:.1%} complete)")
        
        if debug:
            logger.debug(f"Batch {batch_num} - DataFrame sent to OpenAI:\n{batch_df_}")
        
        try:
            product_result = ai_analyze_comments(
                client=client,
                prompt=product_prompt,
                df=batch_df_,
                model=model,
                debug=debug
            )
            
            if debug:
                logger.debug(f"Batch {batch_num} - Raw OpenAI response type: {type(product_result)}")
                logger.debug(f"Batch {batch_num} - Raw OpenAI response length: {len(product_result) if isinstance(product_result, str) else 'N/A'}")
                logger.debug(f"Batch {batch_num} - Raw OpenAI response:\n{product_result}")
            
            cleaned_result = strip_code_block(product_result)
            product_result_df = pd.read_json(StringIO(cleaned_result), orient="records")
            
            product_batches.append(product_result_df)
            
            if export_csv:
                batch_csv = os.path.join(output_dir, f"product_batch{batch_num}.csv")
                product_result_df.to_csv(batch_csv, index=False)
                logger.info(f"Saved product batch {batch_num} to {batch_csv}")
                
        except Exception as e:
            error_msg = f"Exception for product batch {batch_num}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            if 'product_result' in locals() and isinstance(product_result, str):
                error_file = f"failed_product_batch_{batch_num}.json"
                with open(error_file, "w", encoding="utf-8") as f:
                    f.write(product_result)
                logger.error(f"Failed JSON string saved to {error_file}")
            
            # Continue with the next batch instead of raising and stopping
            print(f"Error in product batch {batch_num}, continuing with next batch...")
            continue

    # Combine all product batches
    product_df = pd.concat(product_batches, ignore_index=True) if product_batches else pd.DataFrame()
    
    if export_csv:
        product_combined_csv = os.path.join(output_dir, "product_all_batches_combined.csv")
        product_df.to_csv(product_combined_csv, index=False)
        logger.info(f"Saved combined product results to {product_combined_csv}")
        print(f"Saved combined product results to {product_combined_csv}")

    # CUSTOMER FEEDBACK ANALYSIS
    logger.info("\n===== CUSTOMER FEEDBACK ANALYSIS =====")
    print("\n===== CUSTOMER FEEDBACK ANALYSIS =====")
    customer_batches = []
    
    for batch_num, batch_df_ in batch_df(comments_df, batch_size):
        logger.info(f"Processing customer batch {batch_num}/{total_batches} with {len(batch_df_)} records...")
        print(f"Processing customer batch {batch_num}/{total_batches} with {len(batch_df_)} records... ({batch_num/total_batches:.1%} complete)")
        
        if debug:
            logger.debug(f"Batch {batch_num} - DataFrame sent to OpenAI:\n{batch_df_}")
        
        try:
            customer_result = ai_analyze_comments(
                client=client,
                prompt=customer_prompt,
                df=batch_df_,
                model=model,
                debug=debug
            )
            
            if debug:
                logger.debug(f"Batch {batch_num} - Raw OpenAI response type: {type(customer_result)}")
                logger.debug(f"Batch {batch_num} - Raw OpenAI response length: {len(customer_result) if isinstance(customer_result, str) else 'N/A'}")
                logger.debug(f"Batch {batch_num} - Raw OpenAI response:\n{customer_result}")
            
            cleaned_result = strip_code_block(customer_result)
            customer_result_df = pd.read_json(StringIO(cleaned_result), orient="records")
            
            customer_batches.append(customer_result_df)
            
            if export_csv:
                batch_csv = os.path.join(output_dir, f"customer_batch{batch_num}.csv")
                customer_result_df.to_csv(batch_csv, index=False)
                logger.info(f"Saved customer batch {batch_num} to {batch_csv}")
                
        except Exception as e:
            error_msg = f"Exception for customer batch {batch_num}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            if 'customer_result' in locals() and isinstance(customer_result, str):
                error_file = f"failed_customer_batch_{batch_num}.json"
                with open(error_file, "w", encoding="utf-8") as f:
                    f.write(customer_result)
                logger.error(f"Failed JSON string saved to {error_file}")
            
            # Continue with the next batch instead of raising and stopping
            print(f"Error in customer batch {batch_num}, continuing with next batch...")
            continue

    # Combine all customer batches
    customer_df = pd.concat(customer_batches, ignore_index=True) if customer_batches else pd.DataFrame()
    
    if export_csv:
        customer_combined_csv = os.path.join(output_dir, "customer_all_batches_combined.csv")
        customer_df.to_csv(customer_combined_csv, index=False)
        logger.info(f"Saved combined customer results to {customer_combined_csv}")
        print(f"Saved combined customer results to {customer_combined_csv}")

    # Create a combined CSV of both analyses if requested
    combined_df = None
    if export_csv and len(product_df) > 0 and len(customer_df) > 0:
        combined_csv = "combined_product_customer_analysis.csv"
        combined_data = {
            'row_id': product_df['row_id'] if 'row_id' in product_df.columns else None
        }
        
        # Add comment column without suffix if it exists in either DataFrame
        if 'comment' in product_df.columns:
            combined_data['comment'] = product_df['comment']
        elif 'comment' in customer_df.columns:
            combined_data['comment'] = customer_df['comment']
        
        if 'return_comment' in product_df.columns:
            combined_data['return_comment'] = product_df['return_comment']
        elif 'return_comment' in customer_df.columns:
            combined_data['return_comment'] = customer_df['return_comment']
        
        # Add product columns with _p suffix (excluding comment/return_comment)
        for col in product_df.columns:
            if col != 'row_id' and col != 'comment' and col != 'return_comment':
                combined_data[f"{col}_p"] = product_df[col]
                
        # Add customer columns with _c suffix (excluding comment/return_comment)
        for col in customer_df.columns:
            if col != 'row_id' and col != 'comment' and col != 'return_comment':
                combined_data[f"{col}_c"] = customer_df[col]
        
        combined_df = pd.DataFrame(combined_data)
        combined_df.to_csv(combined_csv, index=False)
        logger.info(f"Saved combined product and customer results to {combined_csv}")
        print(f"Saved combined product and customer results to {combined_csv}")
    elif export_csv:
        logger.warning("Not creating combined CSV file because one or both analysis results are empty")
        print("Not creating combined CSV file because one or both analysis results are empty")

    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    print(f"Analysis completed in {elapsed_time:.2f} seconds")

    return product_df, customer_df, combined_df


def save_sentiment_to_duckdb(
    con, analyzed_product_df, analyzed_customer_df, tname, replace=True, combined_df=None
):
    """
    Join product and customer sentiment results, label columns with _p and _c, and join back to base data.
    Save as {tname}_with_sentiment in DuckDB.
    
    If combined_df is provided, use it instead of recreating the combined dataframe.
    """
    # Check if we have data to save
    if len(analyzed_product_df) == 0 or len(analyzed_customer_df) == 0:
        logger.warning("No data to save to DuckDB - one or both analysis results are empty")
        print("No data to save to DuckDB - one or both analysis results are empty")
        return
    
    # Check if row_id in analyzed data has alphabetic prefixes (like 'A1', 'A2')
    # If so, extract the numeric part for joining
    row_id = 'row_id'
    numeric_id_col = None
    
    # Log DataFrame shapes and columns
    logger.info(f"Product DataFrame shape: {analyzed_product_df.shape}, columns: {analyzed_product_df.columns.tolist()}")
    logger.info(f"Customer DataFrame shape: {analyzed_customer_df.shape}, columns: {analyzed_customer_df.columns.tolist()}")
    
    # Verify row_id column exists in both DataFrames
    if row_id not in analyzed_product_df.columns:
        logger.error(f"row_id column missing from product DataFrame. Available columns: {analyzed_product_df.columns.tolist()}")
        print(f"Error: row_id column missing from product DataFrame")
        return
        
    if row_id not in analyzed_customer_df.columns:
        logger.error(f"row_id column missing from customer DataFrame. Available columns: {analyzed_customer_df.columns.tolist()}")
        print(f"Error: row_id column missing from customer DataFrame")
        return
    
    # Add logging for data inspection
    if len(analyzed_product_df) > 0:
        logger.info(f"Product DataFrame row_id type: {type(analyzed_product_df[row_id].iloc[0])} and dtype: {analyzed_product_df[row_id].dtype}")
        logger.info(f"First few product row_ids: {analyzed_product_df[row_id].head(3).tolist()}")
    else:
        logger.info("Product DataFrame is empty")
        
    if len(analyzed_customer_df) > 0:
        logger.info(f"Customer DataFrame row_id type: {type(analyzed_customer_df[row_id].iloc[0])} and dtype: {analyzed_customer_df[row_id].dtype}")
        logger.info(f"First few customer row_ids: {analyzed_customer_df[row_id].head(3).tolist()}")
    else:
        logger.info("Customer DataFrame is empty")
    
    try:
        # Make copies to avoid modifying the original DataFrames
        analyzed_product_df = analyzed_product_df.copy()
        analyzed_customer_df = analyzed_customer_df.copy()
        
        # Handle null values in row_id columns
        if analyzed_product_df[row_id].isna().any():
            null_count = analyzed_product_df[row_id].isna().sum()
            logger.warning(f"Found {null_count} null values in product DataFrame row_id column")
            analyzed_product_df = analyzed_product_df.dropna(subset=[row_id])
            
        if analyzed_customer_df[row_id].isna().any():
            null_count = analyzed_customer_df[row_id].isna().sum()
            logger.warning(f"Found {null_count} null values in customer DataFrame row_id column")
            analyzed_customer_df = analyzed_customer_df.dropna(subset=[row_id])
        
        # Ensure row_id is string type for both DataFrames
        if len(analyzed_product_df) > 0:
            analyzed_product_df[row_id] = analyzed_product_df[row_id].astype(str)
        if len(analyzed_customer_df) > 0:
            analyzed_customer_df[row_id] = analyzed_customer_df[row_id].astype(str)
            
        logger.info(f"After conversion - Product DataFrame row_id sample: {analyzed_product_df[row_id].iloc[0] if len(analyzed_product_df) > 0 else 'Empty'}")
        logger.info(f"After conversion - Customer DataFrame row_id sample: {analyzed_customer_df[row_id].iloc[0] if len(analyzed_customer_df) > 0 else 'Empty'}")
        
        # Examine row_id formats and determine if we need special handling
        has_non_numeric = False
        
        if len(analyzed_product_df) > 0:
            # Check if any row_id contains non-numeric characters
            has_non_numeric = any(not str(x).isdigit() for x in analyzed_product_df[row_id] if pd.notna(x))
            logger.info(f"Product row_ids have non-numeric characters: {has_non_numeric}")
            
            # More detailed logging about the row_id format
            if has_non_numeric:
                non_numeric_examples = [x for x in analyzed_product_df[row_id].head(10) if not str(x).isdigit()]
                logger.info(f"Examples of non-numeric row_ids: {non_numeric_examples}")
                
                # Check for specific patterns (like 'A1', 'P123', etc.)
                has_alpha_prefix = any(str(x)[0].isalpha() for x in analyzed_product_df[row_id] if pd.notna(x) and len(str(x)) > 0)
                logger.info(f"Row_ids have alphabetic prefix: {has_alpha_prefix}")
        
        # If non-numeric row_ids are detected, extract numeric part
        if has_non_numeric:
            logger.info("Extracting numeric part from row_ids with non-numeric characters")
            numeric_id_col = 'numeric_row_id'
            
            if len(analyzed_product_df) > 0:
                analyzed_product_df[numeric_id_col] = analyzed_product_df[row_id].str.extract(r'([0-9]+)', expand=False)
                logger.info(f"Sample product row_id to numeric_row_id conversion: {list(zip(analyzed_product_df[row_id].head(3), analyzed_product_df[numeric_id_col].head(3)))}")
                
                # Check for extraction failures
                failed_extractions = analyzed_product_df[analyzed_product_df[numeric_id_col].isna()][row_id].head(5).tolist()
                if failed_extractions:
                    logger.warning(f"Failed to extract numeric part from some row_ids: {failed_extractions}")
                
            if len(analyzed_customer_df) > 0:
                analyzed_customer_df[numeric_id_col] = analyzed_customer_df[row_id].str.extract(r'([0-9]+)', expand=False)
                logger.info(f"Sample customer row_id to numeric_row_id conversion: {list(zip(analyzed_customer_df[row_id].head(3), analyzed_customer_df[numeric_id_col].head(3)))}")
            
            # Ensure numeric_row_id is not None and remove rows with failed extraction
            if len(analyzed_product_df) > 0:
                na_count_before = len(analyzed_product_df)
                analyzed_product_df = analyzed_product_df.dropna(subset=[numeric_id_col])
                na_count_after = len(analyzed_product_df)
                if na_count_before > na_count_after:
                    logger.warning(f"Dropped {na_count_before - na_count_after} product rows with failed numeric extraction")
                
            if len(analyzed_customer_df) > 0:
                na_count_before = len(analyzed_customer_df)
                analyzed_customer_df = analyzed_customer_df.dropna(subset=[numeric_id_col])
                na_count_after = len(analyzed_customer_df)
                if na_count_before > na_count_after:
                    logger.warning(f"Dropped {na_count_before - na_count_after} customer rows with failed numeric extraction")
                    
            # Log sample values for debugging
            if len(analyzed_product_df) > 0:
                logger.info(f"Sample row_id: '{analyzed_product_df[row_id].iloc[0]}' -> numeric part: '{analyzed_product_df[numeric_id_col].iloc[0]}'")
            logger.info(f"Using numeric_row_id for database joins")
    
    except Exception as e:
        logger.error(f"Error processing row_id formats: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log sample values for debugging
        if len(analyzed_product_df) > 0:
            logger.error(f"Product DataFrame row_id sample: {analyzed_product_df[row_id].head(3).tolist()}")
            logger.error(f"Product DataFrame row_id dtype: {analyzed_product_df[row_id].dtype}")
        
        if len(analyzed_customer_df) > 0:
            logger.error(f"Customer DataFrame row_id sample: {analyzed_customer_df[row_id].head(3).tolist()}")
            logger.error(f"Customer DataFrame row_id dtype: {analyzed_customer_df[row_id].dtype}")
        
        # Continue without the numeric ID extraction
        logger.info("Continuing with original row_id values without extraction")
        numeric_id_col = None
        
    # Rename columns for product and customer
    try:
        def rename_cols(df, suffix):
            rename_map = {}
            for col in df.columns:
                # Don't rename row_id, numeric_id_col, comment, or return_comment
                if col != row_id and col != numeric_id_col and col != 'comment' and col != 'return_comment':
                    rename_map[col] = f"{col}_{suffix}"
            return df.rename(columns=rename_map)
        
        product_df = rename_cols(analyzed_product_df, 'p')
        customer_df = rename_cols(analyzed_customer_df, 'c')
        
        logger.info(f"Product columns after renaming: {product_df.columns.tolist()}")
        logger.info(f"Customer columns after renaming: {customer_df.columns.tolist()}")
        
        # Use provided combined_df if available, otherwise merge product and customer DataFrames
        if combined_df is not None and len(combined_df) > 0:
            logger.info(f"Using provided combined DataFrame with shape: {combined_df.shape}")
        else:
            # Merge product and customer on row_id
            logger.info(f"Merging product and customer DataFrames on '{row_id}'")
            combined_df = pd.merge(product_df, customer_df, on=row_id, how='outer')
            
            # Handle duplicate comment/return_comment columns if they exist
            for col in ['comment', 'return_comment']:
                p_col = f"{col}_p"
                c_col = f"{col}_c"
                
                # Check if we have both product and customer versions of the column
                if p_col in combined_df.columns and c_col in combined_df.columns:
                    # Keep product version, drop customer version
                    combined_df[col] = combined_df[p_col].combine_first(combined_df[c_col])
                    combined_df = combined_df.drop(columns=[p_col, c_col])
                    logger.info(f"Combined {p_col} and {c_col} into a single {col} column")
                elif p_col in combined_df.columns:
                    # Rename product version to remove suffix
                    combined_df[col] = combined_df[p_col]
                    combined_df = combined_df.drop(columns=[p_col])
                    logger.info(f"Renamed {p_col} to {col}")
                elif c_col in combined_df.columns:
                    # Rename customer version to remove suffix
                    combined_df[col] = combined_df[c_col]
                    combined_df = combined_df.drop(columns=[c_col])
                    logger.info(f"Renamed {c_col} to {col}")
            
            logger.info(f"Merged DataFrame shape: {combined_df.shape}")
        
        if len(combined_df) == 0:
            logger.error("Merge resulted in empty DataFrame - check row_id values in both DataFrames")
            print("Error: Merge resulted in empty DataFrame")
            return
            
        # Log sample of merged data
        logger.info(f"First few rows of merged data:\n{combined_df.head(2).to_string()}")
        
        # Register combined sentiment table
        combined_table_name = f"{tname}_sentiment_combined"
        con.register(combined_table_name, combined_df)
        
        try:
            if replace:
                con.execute(f"CREATE OR REPLACE TABLE {combined_table_name} AS SELECT * FROM {combined_table_name}")
            else:
                con.execute(f"CREATE TABLE IF NOT EXISTS {combined_table_name} AS SELECT * FROM {combined_table_name}")
            logger.info(f"Created combined sentiment table: {combined_table_name}")
        except Exception as e:
            logger.error(f"Error creating combined sentiment table: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error creating combined sentiment table: {str(e)}")
            return
        finally:
            con.unregister(combined_table_name)
    
    except Exception as e:
        logger.error(f"Error during column renaming or DataFrame merge: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Error preparing data for database: {str(e)}")
        return
    
    # Join back to base data
    try:
        # First verify that the base table exists
        base_table_exists = con.execute(f"""
            SELECT count(*) 
            FROM information_schema.tables 
            WHERE table_name = '{tname}'
        """).fetchone()[0]
        
        if not base_table_exists:
            logger.error(f"Base table '{tname}' does not exist in the database")
            print(f"Error: Base table '{tname}' does not exist")
            return
            
        # Check table structure to understand columns
        table_columns = con.execute(f"PRAGMA table_info('{tname}')").fetchdf()
        logger.info(f"Base table columns: {', '.join(table_columns['name'])}")
        
        # Verify row_id column exists in base table
        if 'row_id' not in table_columns['name'].values:
            logger.error(f"row_id column missing from base table '{tname}'")
            print(f"Error: row_id column missing from base table")
            return
        
        # Check combined table exists
        combined_table_exists = con.execute(f"""
            SELECT count(*) 
            FROM information_schema.tables 
            WHERE table_name = '{tname}_sentiment_combined'
        """).fetchone()[0]
        
        if not combined_table_exists:
            logger.error(f"Combined sentiment table '{tname}_sentiment_combined' does not exist")
            print(f"Error: Combined sentiment table does not exist")
            return
            
        combined_table_name = f"{tname}_sentiment_combined"
        
        if numeric_id_col and numeric_id_col in combined_df.columns:
            # Use the numeric part for joining with the database
            logger.info(f"Joining base data using numeric_row_id column")
            logger.info(f"Sample numeric_row_id values: {combined_df[numeric_id_col].head(3).tolist()}")
            
            # Check if numeric_row_id values are valid integers
            try:
                # Test conversion to ensure values can be cast to integers
                sample_values = combined_df[numeric_id_col].head(10).dropna()
                for val in sample_values:
                    int(val)  # This will raise ValueError if conversion fails
                logger.info("Numeric row_id values can be cast to integers")
            except ValueError as e:
                logger.error(f"Error casting numeric_row_id to integer: {str(e)}")
                logger.error(f"Problem values: {sample_values.tolist()}")
                # Fall back to string comparison
                logger.info("Falling back to string comparison for join")
                join_sql = f"""
                    SELECT base.*, sent.*
                    FROM {tname} AS base
                    LEFT JOIN {combined_table_name} AS sent
                    ON CAST(base.row_id AS VARCHAR) = sent.{numeric_id_col}
                """
            else:
                # Use integer casting for the join
                join_sql = f"""
                    SELECT base.*, sent.*
                    FROM {tname} AS base
                    LEFT JOIN {combined_table_name} AS sent
                    ON base.row_id = CAST(sent.{numeric_id_col} AS INTEGER)
                """
        else:
            # Standard join on row_id
            logger.info(f"Joining base data using standard row_id column")
            if len(combined_df) > 0:
                logger.info(f"Sample row_id values: {combined_df[row_id].head(3).tolist()}")
                
            join_sql = f"""
                SELECT base.*, sent.*
                FROM {tname} AS base
                LEFT JOIN {combined_table_name} AS sent
                ON CAST(base.row_id AS VARCHAR) = CAST(sent.row_id AS VARCHAR)
            """
        
        # Log the SQL query for debugging
        logger.info(f"Executing join SQL: {join_sql}")
        
        # Execute the join
        joined_df = con.execute(join_sql).fetchdf()
        logger.info(f"Join completed. Result has {len(joined_df)} rows and {len(joined_df.columns)} columns")
        
        # Check if join produced expected results
        if len(joined_df) == 0:
            logger.warning("Join resulted in empty DataFrame - check join conditions")
            
        # Check for column duplication
        duplicate_cols = joined_df.columns.duplicated()
        if any(duplicate_cols):
            dup_col_names = joined_df.columns[duplicate_cols].tolist()
            logger.warning(f"Join resulted in duplicate column names: {dup_col_names}")
            
            # Handle duplicate columns by renaming
            logger.info("Renaming duplicate columns to avoid conflicts")
            new_columns = []
            seen = set()
            
            for col in joined_df.columns:
                if col in seen:
                    count = 1
                    new_col = f"{col}_{count}"
                    while new_col in seen:
                        count += 1
                        new_col = f"{col}_{count}"
                    new_columns.append(new_col)
                    seen.add(new_col)
                else:
                    new_columns.append(col)
                    seen.add(col)
                    
            joined_df.columns = new_columns
        
        # Save joined result as a new table
        joined_table_name = f"{tname}_with_sentiment"
        con.register(joined_table_name, joined_df)
        
        try:
            if replace:
                con.execute(f"CREATE OR REPLACE TABLE {joined_table_name} AS SELECT * FROM {joined_table_name}")
            else:
                con.execute(f"CREATE TABLE IF NOT EXISTS {joined_table_name} AS SELECT * FROM {joined_table_name}")
            
            # Check if table was created successfully
            table_exists = con.execute(f"""
                SELECT count(*) 
                FROM information_schema.tables 
                WHERE table_name = '{joined_table_name}'
            """).fetchone()[0]
            
            if table_exists:
                logger.info(f"Successfully created joined table '{joined_table_name}' with {len(joined_df)} rows")
                print(f"Joined product and customer sentiment results saved to DuckDB as '{joined_table_name}'.")
            else:
                logger.error(f"Failed to create joined table '{joined_table_name}'")
                print(f"Error: Failed to create joined table")
                
        except Exception as e:
            logger.error(f"Error creating joined table: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error creating joined table: {str(e)}")
        finally:
            con.unregister(joined_table_name)
        
    except Exception as e:
        logger.error(f"Error during database join operation: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Error saving results to database: {str(e)}")
        # Don't re-raise the exception to allow the script to complete
        return


def main():
    """Main function to run the sentiment analyzer."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load config file if it exists
    config = load_config(args.config)
    
    # Combine config and command line arguments, with command line taking precedence
    file_path = args.file_path if args.file_path is not None else config.get('file_path')
    table_name = args.table_name if args.table_name is not None else config.get('table_name')
    db_path = args.db_path if args.db_path is not None else config.get('db_path')
    row_id = args.row_id if args.row_id is not None else config.get('row_id', 'row_id')
    comment_column = args.comment_column if args.comment_column is not None else config.get('comment_column', 'RETURN_COMMENT')
    
    # For boolean flags, we need special handling because argparse's store_true action makes them False by default
    # For command line args: Explicitly provided = True, not provided = False
    # For config values: Use what's in the config or default to False
    if args.no_sample:
        # --no_sample flag takes precedence
        is_sample = False
    elif args.sample:
        # --sample flag was provided
        is_sample = True
    else:
        # Check config or default to False
        is_sample = config.get('sample', False)
    
    clear_data = args.clear_data or config.get('clear_data', False)
    
    sample_size = args.sample_size if args.sample_size is not None else config.get('sample_size', 500)
    batch_size = args.batch_size if args.batch_size is not None else config.get('batch_size', 750)
    batch_size = args.batch_size if args.batch_size is not None else config.get('batch_size', 750)
    # For boolean flags, similar handling as above
    if args.import_data:
        import_data_flag = True
    else:
        import_data_flag = config.get('import_data', True)
        
    if args.skip_analyzed:
        skip_analyzed = True
    elif args.force_reanalyze:
        skip_analyzed = False
    else:
        skip_analyzed = config.get('skip_analyzed', True)
        
    if args.debug:
        debug = True
    else:
        debug = config.get('debug', False)
        
    if args.export_csv:
        export_csv = True
    else:
        export_csv = config.get('export_csv', True)
    
    # Non-boolean argument
    model = args.model if args.model is not None else config.get('model', 'gpt-4o')
    
    # Validate required parameters
    if not file_path:
        file_path = input("Enter path to input file: ")
    if not table_name:
        table_name = input("Enter table name to create: ")
    if not db_path:
        db_path = input("Enter path to DuckDB database: ")
    
    # Log start of processing
    logger.info("Starting sentiment analysis process")
    logger.info(f"Input file: {file_path}")
    logger.info(f"Table name: {table_name}")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Using sample: {is_sample} (Value from config.json: {config.get('sample', False)})")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Model: {model}")
    logger.info(f"Import data: {import_data_flag}")
    logger.info(f"Skip already analyzed records: {skip_analyzed}")
    if args.force_reanalyze:
        logger.info("Forcing reanalysis of all records, including previously analyzed ones")
    
    print(f"Using sample: {is_sample} (Value from config.json: {config.get('sample', False)})")
    print(f"Sample size: {sample_size}")
    if args.force_reanalyze:
        print("Forcing reanalysis of all records, including previously analyzed ones")
    
    # Import data into DuckDB
    if import_data_flag:
        logger.info(f"Importing data from {file_path} to {db_path}.{table_name}")
        import_data(file_path, clear=clear_data, db_path=db_path, tname=table_name)
    else:
        logger.info(f"Skipping data import as requested")
    
    # Connect to DuckDB
    con = duckdb.connect(db_path)
    
    # Fetch comments data
    comments_df = fetch_return_comments(
        con, 
        table_name, 
        is_sample=is_sample,
        sample_size=sample_size,
        comment_column=comment_column, 
        row_id_column=row_id,
        skip_analyzed=skip_analyzed
    )
    
    # Check if we have any comments to process
    if len(comments_df) == 0:
        if skip_analyzed:
            logger.info("No new comments found to analyze. All records in the database have already been analyzed.")
            print("No new comments found to analyze. All records in the database have already been analyzed.")
        else:
            logger.warning("No comments found to analyze. Check your filters and database.")
            print("No comments found to analyze. Check your filters and database.")
        con.close()
        return
    
    # Access OpenAI API key from GCP Secret Manager
    try:
        secret = access_secret("projects/572292574132/secrets/openai_monday_status_alerts/versions/latest")
        openai_client = openai.OpenAI(api_key=secret)
    except Exception as e:
        logger.error(f"Failed to access OpenAI API key: {e}")
        print(f"Failed to access OpenAI API key: {e}")
        return
    
    # Run sentiment analysis
    product_df, customer_df, combined_df = handle_sentiment_analysis(
        comments_df,
        client=openai_client,
        product_prompt_path='prompts/product_prompt.txt',
        customer_prompt_path='prompts/customer_sentiment_prompt.txt',
        batch_size=batch_size,
        debug=debug,
        export_csv=export_csv,
        id_column=row_id,
        model=model,
    )
    
    # Save results to DuckDB if both product and customer dataframes have data
    if len(product_df) > 0 and len(customer_df) > 0:
        save_sentiment_to_duckdb(con, product_df, customer_df, table_name, replace=True, combined_df=combined_df)
    else:
        logger.warning("Not saving to DuckDB - one or both analysis results are empty")
        print("Not saving to DuckDB - one or both analysis results are empty")
    
    # Close connection
    con.close()
    
    logger.info("Sentiment analysis process completed")
    print("Sentiment analysis process completed")


if __name__ == "__main__":
    main()
