"""
This script is a more-production ready version of the main.ipynb notebook pipeline.
Key functions:

Import_data: Load CSV, Parquet, or Excel files into a DuckDB table.
fetch_return_comments: Extract return comments from the DuckDB table.
access_secret: Access secrets from Google Cloud Secret Manager.
ai_analyze_comments: Send comments to OpenAI for sentiment analysis.
export_results: Export results to Excel and CSV files.
run_sentiment_pipeline: Handler function to run the entire sentiment analysis pipeline.
"""

import os
import pandas as pd
import duckdb
import json
from typing import Optional, Tuple
import openai


# --- Data Import ---
def import_data(fname: str, db_path: str = 'temp_db', tname: str = 'staging_table', clear: bool = True, ftype: Optional[str] = None):
    """
    load CSV, Parquet, or Excel file into a DuckDB table.
    Returns confirmation message.
    clear = Replace existing table if true, else create if not exists.
    """
    con = duckdb.connect(db_path)
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
        con.execute(f"""
            CREATE {mode} {tname} AS
            SELECT * FROM read_csv_auto('{fname}', escape='\\', encoding='utf-8', header=True)
        """)
    elif ftype == 'parquet':
        mode = 'OR REPLACE TABLE' if clear else 'TABLE IF NOT EXISTS'
        con.execute(f"""
            CREATE {mode} {tname} AS
            SELECT * FROM read_parquet('{fname}')
        """)
    elif ftype == 'excel':
        df = pd.read_excel(fname)
        if clear:
            con.execute(f"DROP TABLE IF EXISTS {tname}")
        con.register('temp_excel_df', df)
        con.execute(f"CREATE TABLE {tname} AS SELECT * FROM temp_excel_df")
        con.unregister('temp_excel_df')
    else:
        con.close()
        raise ValueError("Unsupported file type.")
    return con

# --- Data Extraction ---
def fetch_return_comments(con, tname, is_sample: bool = False, comment_only: bool = True) -> pd.DataFrame:
    """
    Fetch return comments from the DuckDB table.
    If is_sample is True, fetch a sample of 100 rows.
    If comment_only is True, only returns the RETURN_COMMENT column.
    If comment_only is False, returns all columns.
    """
    sample_query = "ORDER BY RANDOM() LIMIT 100" if is_sample else ""
    if comment_only:
        query = f'''
        SELECT "RETURN COMMENT"
        FROM {tname}
        {sample_query}
        '''
    else:
        query = f'''
        SELECT *
        FROM {tname}
        {sample_query}
        '''
    return con.execute(query).df()

# --- Secret Manager ---
def access_secret(secret_path):
    """Establishes connection to GCP secret manager and retrieves secret value.
    ensure authentication is setup for GCP: in bash: gcloud auth application-default login"""
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(name=secret_path)
    secret_payload = response.payload.data.decode("UTF-8")
    return secret_payload

# --- Sentiment Analysis ---
def ai_analyze_comments(client, prompt: str, df: pd.DataFrame, debug: bool = True) -> str:
    """
    Sends `prompt` plus the JSON version of `df` to ChatGPT,
    and returns the model's response.strip()
    """
    df_json = df.to_json(orient="records")
    if debug:
        print("Prompt sent to model:\n", prompt)
    messages = [
        {"role": "system", "content": (
            "You are an expert linguistic analyst specializing in extracting and scoring themes from customer return comments. "
            "You always return your output as a single JSON array of objects, one per input record, using exactly the keys and structure specified in the user's instructions. "
            "Do not include any explanations, extra text, or formatting outside the required JSON array. "
            "Be precise, consistent, and strictly follow the output schema and scoring rules provided."
        )},
        {"role": "user", "content": prompt},
        {"role": "user", "content": df_json}
    ]
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        temperature=0.1,
    )
    content = resp.choices[0].message.content.strip()
    if debug:
        print("Raw response from OpenAI:\n", content)
    if not content:
        raise ValueError("Empty response from OpenAI")
    return content

# --- Export Results ---
def export_results(df: pd.DataFrame, base_name: str = "analyzed_feedback_output", excel: bool = True, csv: bool = True, debug: bool = True):
    """
    Exports the response string to a CSV file if it is valid JSON,
    otherwise saves it as a TXT file.
    """
    
    if excel:
        try:
            out_xlsx = f"{base_name}.xlsx"
            df.to_excel(out_xlsx, index=False)
            if debug:
                print(f"Exported results to {out_xlsx}")
        except Exception as e:
            print(f"Failed to export to Excel: {e}")
    if csv:
        try:
            out_csv = f"{base_name}.csv"
            df.to_csv(out_csv, index=False)
            if debug:
                print(f"Exported results to {out_csv}")
        except Exception as e:
            print(f"Failed to export to CSV: {e}")

# --- Main Pipeline Function ---
def run_sentiment_pipeline(
    data_path: Optional[str] = None,
    db_path: str = 'temp_db',
    tname: str = 'staging_table',
    customer_prompt_path: str = 'prompts/customer_sentiment_prompt.txt',
    product_prompt_path: str = 'prompts/product_prompt.txt',
    secret_path: Optional[str] = None,
    batch_size: int = 100,
    debug: bool = True,
    excel_export: bool = True,
    csv_export: bool = True,
    con: duckdb.DuckDBPyConnection = None,
    df: pd.DataFrame = None
):
    """
    Run the full sentiment analysis pipeline on customer feedback data and product feedback data.

    Steps:
    1. Data Import (optional):
        - If `df` is provided, use it directly as the input DataFrame.
        - If `con` (a DuckDB connection) is provided, extract data from the specified table (`tname`).
        - If `data_path` is provided, import the file (CSV, Excel, or Parquet) into DuckDB and extract data from the table.
        - If none of these are provided, the function will exit with a message.
    2. API Key Setup:
        - If `secret_path` is provided, fetch the OpenAI API key from Google Secret Manager.
        - Otherwise, use the `OPENAI_API_KEY` or `P_OPEN_API_KEY` environment variable.
    3. Prompt Loading:
        - Load the customer and product prompt text from the respective files.
    4. Sentiment Analysis:
        - For both customer and product prompts, process the DataFrame in batches.
        - For each batch, send the data and prompt to the OpenAI API and collect the results.
    5. Export Results:
        - Export the customer and product results to Excel and/or CSV, depending on `excel_export` and `csv_export` flags.
        - If `excel_export` is True, combine both results into a single Excel file with two sheets.

    Parameters:
        data_path (str, optional): Path to the input data file (CSV, Excel, or Parquet). If None, must provide `df` or `con`.
        db_path (str): Path to the DuckDB database file.
        tname (str): Name of the DuckDB table to use.
        customer_prompt_path (str): Path to the customer sentiment prompt file.
        product_prompt_path (str): Path to the product feedback prompt file.
        secret_path (str, optional): GCP Secret Manager path for the OpenAI API key.
        batch_size (int): Number of rows to process per API call.
        debug (bool): If True, print debug information.
        excel_export (bool): If True, export results to an Excel file.
        csv_export (bool): If True, export results to a CSV file.
        con (duckdb.DuckDBPyConnection, optional): Existing DuckDB connection to use for data extraction.
        df (pd.DataFrame, optional): DataFrame to use directly as input.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (customer_results, product_results)
    """
    # Import data if data_path is provided and df is not
    if df is not None:
        pass  # Use provided DataFrame
    elif con is not None:
        df = fetch_return_comments(con, tname, is_sample=False, comment_only=True)
    elif data_path is not None:
        con = import_data(data_path, db_path=db_path, tname=tname, clear=True)
        df = fetch_return_comments(con, tname, is_sample=False, comment_only=True)
    else:
        print("No data source provided. Please provide data_path, con, or df.")
        return None, None
    # Get OpenAI API key
    if secret_path:
        api_key = access_secret(secret_path)
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("P_OPEN_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    # Load prompts
    with open(customer_prompt_path, 'r', encoding='utf-8') as f:
        customer_prompt = f.read()
    with open(product_prompt_path, 'r', encoding='utf-8') as f:
        product_prompt = f.read()
    # Helper for batch processing
    def batch_process(prompt, df, label):
        results = []
        total = len(df)
        if total == 0:
            print(f"Input DataFrame for {label} is empty.")
            return None
        if total <= batch_size:
            if debug:
                print(f"Processing all {total} rows in a single batch for {label}.")
            try:
                response_str = ai_analyze_comments(client, prompt, df, debug=debug)
                batch_results = json.loads(response_str)
                if isinstance(batch_results, dict):
                    batch_results = [batch_results]
                results.extend(batch_results)
            except Exception as e:
                print(f"Error processing single batch for {label}: {e}")
                return None
        else:
            for i in range(0, total, batch_size):
                batch = df.iloc[i:i+batch_size]
                if debug:
                    print(f"Processing batch {i//batch_size + 1} ({i} to {min(i+batch_size, total)-1}) for {label}")
                try:
                    response_str = ai_analyze_comments(client, prompt, batch, debug=debug)
                    batch_results = json.loads(response_str)
                    if isinstance(batch_results, dict):
                        batch_results = [batch_results]
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error processing batch {i//batch_size + 1} for {label}: {e}")
                    continue
        if not results:
            print(f"No results to export for {label}.")
            return None
        return pd.DataFrame(results)
    # Run both analyses
    customer_results = batch_process(customer_prompt, df, label="Customer Sentiment")
    product_results = batch_process(product_prompt, df, label="Product Feedback")
    # Export
    if excel_export and customer_results is not None and product_results is not None:
        try:
            with pd.ExcelWriter("combined_feedback_analysis.xlsx", engine='openpyxl') as writer:
                product_results.to_excel(writer, sheet_name='Product Feedback', index=False)
                customer_results.to_excel(writer, sheet_name='Customer Feedback', index=False)
            if debug:
                print("Exported combined results to combined_feedback_analysis.xlsx")
        except Exception as e:
            print(f"Failed to export combined Excel file: {e}")
    if csv_export:
        if customer_results is not None:
            export_results(customer_results, base_name="analyzed_customer_feedback_output", excel=False, csv=True, debug=debug)
        if product_results is not None:
            export_results(product_results, base_name="analyzed_product_feedback_output", excel=False, csv=True, debug=debug)
    return customer_results, product_results

if __name__ == "__main__":
    # Example usage
    run_sentiment_pipeline(
        data_path='data/RETURN_COMMENTS_GROUP.xlsx',
        db_path='return_comment_group',
        tname='staging_table',
        customer_prompt_path='prompts/customer_sentiment_prompt.txt',
        product_prompt_path='prompts/product_prompt.txt',
        secret_path=os.getenv("OPEN_API_KEY"),
        batch_size=100,
        debug=True,
        excel_export=True,
        csv_export=True
    )
