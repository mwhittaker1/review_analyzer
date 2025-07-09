import os
import json
import logging
import pandas as pd
import openai
import re
from dotenv import load_dotenv
import duckdb
from typing import Optional, Tuple
import openpyxl

openai.api_key = os.getenv("OPEN_API_KEY")
if not openai.api_key:
    raise EnvironmentError("Please set the OPEN_API_KEY environment variable")


def import_data(
        fname: str,
        clear=True,
        db_path='temp_db',
        tname='staging_table',
        ftype='csv',
        fetch_data: bool = True
        )-> Tuple[duckdb.DuckDBPyConnection, Optional[pd.DataFrame]]:
    """
    load CSV or Parquet file into a DuckDB table.
    fname = Filename of the file to load
    clear = If True, drop the table if it exists before loading, useful for reseting data.
    db_path = Name of the DuckDB database (default: 'temp_db')
    tname = Name of the table to create (default: 'staging_table')
    ftype = Type of file to load ('csv' or 'parquet', default: 'csv')
    Returns a DuckDB connection object.
    """
    con = duckdb.connect(db_path)

    if ftype not in ('csv', 'parquet'):
        con.close()
        raise ValueError("file_type must be 'csv' or 'parquet'")
    
    # handles:
    #   clearing table if clear is True
    #   create table if not exists
    #   import csv and parquet files
    func = 'read_csv_auto' if ftype == 'csv' else 'read_parquet'
    mode = 'OR REPLACE TABLE' if clear else 'TABLE IF NOT EXISTS'
    con.execute(f"""
        CREATE {mode} {tname} AS
        SELECT * FROM {func}('{fname}')
    """)

    # if fetch_data is True, return the data as a DataFrame
    df: Optional[pd.DataFrame] = None
    if fetch_data:
        df = con.execute(f"SELECT * FROM {tname}").df()

    return con, df


def parse_gpt_response(response: str, debug: bool = True) -> dict:
    """
    Parse the GPT response in the custom template format into a dictionary.
    """
    # Remove START/END and split lines
    lines = response.replace('START', '').replace('END', '').strip().split('\n')
    result = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match Theme/Sentiment/Total_Sentiment
        theme_match = re.match(r"Theme (\d+): (.+)", line)
        sentiment_match = re.match(r"Sentiment (\d+): (.+)", line)
        total_match = re.match(r"Total_Sentiment: (.+)", line)
        if theme_match:
            idx, value = theme_match.groups()
            result[f"Theme {idx}"] = value.strip()
        elif sentiment_match:
            idx, value = sentiment_match.groups()
            result[f"Sentiment {idx}"] = value.strip()
        elif total_match:
            value = total_match.group(1).strip()
            result["Total_Sentiment"] = value
    if debug:
        print("Parsed single response block:", result)
    return result

def ai_format_df(prompt: str, df: pd.DataFrame, debug: bool = True, use_case: str = "product") -> pd.DataFrame:
    """
    Sends `prompt` plus the JSON version of `df` to ChatGPT,
    and returns the model's JSON response as a new DataFrame.
    Uses structured JSON output format for more reliable results.
    The `use_case` parameter should be either "product" or "customer".
    """

    df_json = df.to_json(orient="records")

    if debug:
        print("Prompt sent to model:\n", prompt)

    # Set system message based on use_case, matching the intent of each prompt file
    if use_case == "customer":
        system_message = (
            "You are a linguistic expert in customer satisfaction evaluation. "
            "You will receive a JSON array of return comments, each with RETURN_NO and RETURN_COMMENT. "
            "For each record, extract up to four satisfaction themes (1–5 words each) and assign a sentiment score (1–5) for each, "
            "following the scoring rules in the user prompt. "
            "Return your answer as a single JSON array of objects, one per input record, with the exact keys and structure described in the user prompt. "
            "Do not include explanations or extra text—only the JSON output."
        )
    else:  # default to product
        system_message = (
            "You are a linguistic expert in product evaluation. "
            "You will receive a batch of return comments as a JSON array, each with RETURN_NO and RETURN_COMMENT. "
            "For each record, extract up to four product themes and assign a sentiment score (1–5) for each, "
            "following the scoring rules in the user prompt. "
            "Return your answer as a single JSON array of objects, one per input record, with the exact keys and structure described in the user prompt. "
            "Do not include explanations or extra text—only the JSON output."
        )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
        {"role": "user", "content": df_json}
    ]

    # Request JSON format specifically
    resp = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    content = resp.choices[0].message.content.strip()
    if debug:
        print("Raw response from OpenAI:\n", content)
    if not content:
        raise ValueError("Empty response from OpenAI")
    logging.debug(f"AI response: {content}")

    # Parse the JSON response
    try:
        data = json.loads(content)

        # Handle the case where the response might be a single object or list of records
        if isinstance(data, list):
            result_data = data
        elif "records" in data:
            result_data = data["records"]
        elif isinstance(data, dict):
            # Single record or wrapper
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    result_data = value
                    break
            else:
                result_data = [data]
        else:
            result_data = data

        if debug:
            print("Parsed data from response:\n", result_data)

        result_df = pd.DataFrame(result_data)
    except json.JSONDecodeError as e:
        if debug:
            print(f"Response is not valid JSON: {e}")
            print("Falling back to custom parser")
        # Fallback to your custom parser for compatibility
        blocks = [b for b in content.split('END') if b.strip()]
        parsed = [parse_gpt_response(block, debug=debug) for block in blocks]
        if debug:
            print("Parsed data from response (custom):\n", parsed)
        result_df = pd.DataFrame(parsed)

    return result_df


def batch_ai_format_df(prompt, df, batch_size=50, debug=False, use_case="product"):
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size  # ceiling division
    print(f"Processing {len(df)} records in {total_batches} batches of {batch_size}...")
    
    for i in range(0, len(df), batch_size):
        batch_num = i // batch_size + 1
        batch = df.iloc[i:i+batch_size]
        print(f"Processing batch {batch_num}/{total_batches} (records {i+1}-{min(i+batch_size, len(df))})")
        if batch.empty:
            print(f"Batch {batch_num} is empty, skipping.")
            continue
        out = ai_format_df(prompt, batch, debug=debug, use_case=use_case)
        if out is not None and not out.empty:
            results.append(out)
            print(f"Completed batch {batch_num}/{total_batches}, {len(out)} records returned.")
        else:
            print(f"Warning: Batch {batch_num} returned no results.")
    
    if results:
        print(f"All {total_batches} batches processed successfully. Total records: {sum(len(r) for r in results)}")
        return pd.concat(results, ignore_index=True)
    else:
        print("No results to concatenate. Returning empty DataFrame.")
        return pd.DataFrame()


def evaluate_and_store_product(df, prompt, db_path='first_db', table_name='product_evaluated', batch_size=50, debug=False):
    """
    Runs product evaluation on Dataframe 'df'
    Creates/replaces table {table_name} in DuckDB with product evaluation results.
    """
    
    print(f"\n--- Starting Product Evaluation ({len(df)} records) ---")
    product_eval = batch_ai_format_df(prompt, df, batch_size=batch_size, debug=debug, use_case="product")
    print(f"Storing product evaluation results to DuckDB table '{table_name}'...")
    con = duckdb.connect(db_path)
    # Register the DataFrame as a view that DuckDB can query
    con.register('product_eval_view', product_eval)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM product_eval_view")
    con.close()
    print(f"Product evaluation complete: {len(product_eval)} records processed and stored.")
    return product_eval


def evaluate_and_store_customer(df, prompt, db_path='first_db', table_name='customer_evaluated', batch_size=50, debug=False):
    print(f"\n--- Starting Customer Evaluation ({len(df)} records) ---")
    customer_eval = batch_ai_format_df(prompt, df, batch_size=batch_size, debug=debug, use_case="customer")
    print(f"Storing customer evaluation results to DuckDB table '{table_name}'...")
    con = duckdb.connect(db_path)
    # Register the DataFrame as a view that DuckDB can query
    con.register('customer_eval_view', customer_eval)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM customer_eval_view")
    con.close()
    print(f"Customer evaluation complete: {len(customer_eval)} records processed and stored.")
    return customer_eval


def clean_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove illegal characters for Excel from all string columns in the DataFrame.
    """
    import re
    # Excel does not allow control characters except for tab (\x09), line feed (\x0A), and carriage return (\x0D)
    illegal_char_re = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
    def clean_cell(val):
        if isinstance(val, str):
            return illegal_char_re.sub('', val)
        return val
    return df.applymap(clean_cell)


def export_evaluations_to_excel(db_path='first_db', product_table='product_evaluated', customer_table='customer_evaluated', out_path='evaluation_results.xlsx'):
    """
    Export product and customer evaluation tables from DuckDB to a single Excel file with two sheets.
    """
    import duckdb
    import pandas as pd
    with duckdb.connect(db_path) as con:
        product_df = con.execute(f"SELECT * FROM {product_table}").df()
        customer_df = con.execute(f"SELECT * FROM {customer_table}").df()
    # Clean dataframes before export
    product_df = clean_for_excel(product_df)
    customer_df = clean_for_excel(customer_df)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        product_df.to_excel(writer, sheet_name='Product', index=False)
        customer_df.to_excel(writer, sheet_name='Customer', index=False)
    print(f"Exported to {out_path}")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    return_comments_file = r'data/200_comments_id_comment_only.csv'

    # Example usage
    con, df = import_data(return_comments_file, clear=True, db_path='first_db', tname='staging_table', ftype='csv')

    # Define prompts for product and customer evaluations
    with open('product_prompt.txt', 'r', encoding='utf-8') as f:
        product_prompt = f.read()
    with open('customer_sentiment_prompt.txt', 'r', encoding='utf-8') as f:
        customer_prompt = f.read()

    # Evaluate and store results
    product_eval = evaluate_and_store_product(
        df, 
        product_prompt, 
        db_path='first_db', 
        table_name='product_evaluated', 
        batch_size=50, 
        debug=False
    )
    customer_eval = evaluate_and_store_customer(
        df, 
        customer_prompt, 
        db_path='first_db', 
        table_name='customer_evaluated', 
        batch_size=50, 
        debug=False
    )

    # Export results to Excel
    export_evaluations_to_excel(
        db_path='first_db', 
        product_table='product_evaluated', 
        customer_table='customer_evaluated', 
        out_path='results/evaluation_results_200_comments_only_GPT-4.1_nano.xlsx'
    )