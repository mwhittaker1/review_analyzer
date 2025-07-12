# Sentiment Analyzer for Return Comments

This Python script performs sentiment analysis on customer return comments using OpenAI's GPT-4o model. It analyzes both product-related feedback and customer sentiment, then saves the results to a DuckDB database and CSV files.

## Prerequisites

- Python 3.8+
- Required Python packages:
  - pandas
  - duckdb
  - openai
  - google-cloud-secretmanager
  - python-dotenv

## Installation

```bash
pip install pandas duckdb openai google-cloud-secretmanager python-dotenv
```

## Usage

### Basic Usage

Run the script with default settings:

```bash
python sentiment_analyzer_script.py
```

### Command Line Arguments

You can customize the behavior using command line arguments:

```bash
python sentiment_analyzer_script.py --file_path "path/to/data.csv" --table_name "my_table" --db_path "my_database.db"
```

Available arguments:

- `--config`: Path to configuration file (default: config.json)
- `--clear_data`: Clear existing data on import
- `--file_path`: Path to input CSV file
- `--table_name`: Name of the table to create
- `--db_path`: Path to DuckDB database
- `--row_id`: Column to use as unique identifier (default: row_id)
- `--comment_column`: Column containing return comments (default: RETURN_COMMENT)
- `--sample`: Run analysis on a sample of records
- `--sample_size`: Size of sample to analyze (default: 500)
- `--batch_size`: Batch size for API calls (default: 750)
- `--debug`: Enable debug output
- `--export_csv`: Export results to CSV
- `--model`: OpenAI model to use for analysis (default: gpt-4o)
- `--import_data`: Import data from file to database (when false, uses existing database table)
- `--skip_analyzed`: Skip records that already have sentiment analysis
- `--force_reanalyze`: Force reanalysis of already analyzed records (overrides --skip_analyzed)

### Configuration File

You can also use a configuration file (JSON format) to set these parameters:

```json
{
    "file_path": "C:\\Code\\URBN\\review_analyzer\\data\\IID_RR_GROSS.csv",
    "table_name": "iid_return_reasons",
    "db_path": "iid_return_comment.db",
    "row_id": "row_id",
    "comment_column": "RETURN_COMMENT",
    "clear_data": true,
    "sample": false,
    "sample_size": 500,
    "batch_size": 750,
    "debug": false,
    "export_csv": true,
    "model": "gpt-4o",
    "import_data": true,
    "skip_analyzed": true
}
```

## Workflow

1. The script loads data from the specified file into a DuckDB table
2. Comments are extracted and processed in batches
3. Each batch is sent to OpenAI for sentiment analysis (product and customer perspectives)
4. Results are saved as CSV files (both individual batches and combined)
5. Results are also saved back to the DuckDB database

## Prompt Files

The script expects two prompt files in the `prompts` directory:

- `prompts/product_prompt.txt`: Instructions for product-focused analysis
- `prompts/customer_sentiment_prompt.txt`: Instructions for customer sentiment analysis

## Output

- Individual batch CSV files in the `batches` directory
- Combined CSV files: 
  - `batches/product_all_batches_combined.csv`
  - `batches/customer_all_batches_combined.csv`
  - `combined_product_customer_analysis.csv`
- DuckDB tables:
  - `{table_name}_sentiment_combined`: Combined sentiment results
  - `{table_name}_with_sentiment`: Original data joined with sentiment results
- Log file: `sentiment_analysis_run.log`

## Authentication

The script uses Google Cloud Secret Manager to access the OpenAI API key. Make sure you have authenticated with Google Cloud:

```bash
gcloud auth application-default login
```

