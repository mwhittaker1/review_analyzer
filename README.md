# Review Analyzer

A tool for analyzing customer review comments using OpenAI's GPT models and DuckDB.

## Overview

This project provides a system for analyzing customer product review comments. It:

- Extracts key themes from each comment
- Determines sentiment for each theme
- Calculates overall sentiment for each comment
- Stores results in DuckDB
- Exports analysis to Excel and CSV

## Main Features

- Processes large datasets in batches
- Uses OpenAI GPT-4o for analysis
- Supports both customer sentiment and product feedback analysis
- Exports results to combined Excel and CSV files
- Modular pipeline: use interactively (notebook) or as a script

## Key Files

- `main.ipynb` – Interactive notebook for step-by-step analysis and prototyping
- `sentiment_analyzer_script.py` – Script for batch analysis and export
- `enhanced_sentiment_analyzer.py` – Enhanced version that fixes the issue with skipping records
- `prompts/` – Contains prompt templates for different analysis types
- `batch_retry.py` – Helper script for retrying failed batches with smaller chunk sizes

## Recent Enhancements

We've addressed several issues with the sentiment analyzer:

1. **Fixed Record Skipping Issue**: The original script incorrectly skipped all records when checking for already analyzed comments. See `README_sentiment_analyzer_fix.md` for details.
2. **Enhanced Batch Processing**: Added batch retry functionality to handle failures with smaller chunk sizes.
3. **Improved Error Handling**: Created utilities to extract row IDs from failed batches and retry them.

## Recommended Workflow

1. **Use the Enhanced Script**: Use `enhanced_sentiment_analyzer.py` instead of the original script to correctly process all comments.
2. **Use Smaller Batch Sizes**: Set batch_size to 100-250 instead of 500-750 to prevent API response truncation.
3. **Handle Failed Batches**: If a batch fails, use `fix_truncated_json.py` to extract the row IDs, then use `batch_retry.py` to process them with a smaller chunk size.

## Workflow

1. **Import data** (CSV, Excel, or Parquet) into DuckDB
2. **Extract comments** to a DataFrame
3. **Run customer sentiment and product feedback analysis** using OpenAI GPT
4. **Export results**:
   - Combined Excel file with two sheets (Customer Feedback, Product Feedback)
   - Separate CSV files for each analysis

## Usage

### Interactive Analysis

- Open `main.ipynb` for a step-by-step workflow.
- Demonstrates data import, DuckDB usage, prompt engineering, and batch sentiment analysis.

### Automated Batch Analysis

- Use `enhanced_sentiment_analyzer.py` for a scriptable pipeline.
- Run with:
  ```
  python enhanced_sentiment_analyzer.py
  ```

- For advanced configuration, use the config.json file or command line arguments:
  ```
  python enhanced_sentiment_analyzer.py --force_reanalyze --batch_size 100
  ```

### Handling Failed Batches

If a batch fails during processing:

1. Extract row IDs from the failed batch:
   ```
   python fix_truncated_json.py --log_file sentiment_analysis_run.log --batch_num 8
   ```

2. Retry the failed records with a smaller chunk size:
   ```
   python batch_retry.py --row_ids_file batch8_row_ids.txt --chunk_size 50
   ```

## Troubleshooting

If the script reports "No comments found to analyze" but you expect unprocessed comments:

1. Check if the sentiment table has valid scores:
   ```
   python check_sentiment_scores.py
   ```

2. If needed, use the `--force_reanalyze` flag to reprocess all records:
   ```
   python enhanced_sentiment_analyzer.py --force_reanalyze
   ```

See `README_sentiment_analyzer_fix.md` for detailed troubleshooting information.

## Customization

- Modify prompt files in `prompts/` to adjust analysis behavior.

## Output

- Theme columns (Theme 1, Theme 2, ...)
- Sentiment columns (Sentiment 1, Sentiment 2, ...)
- `Total_Sentiment` column for overall sentiment
- Results stored in DuckDB and exported to Excel/CSV

## Setup

1. Set your OpenAI API key in a `.env` file or use GCP Secret Manager:
   ```
   OPEN_API_KEY=your_api_key_here
   ```
2. Install requirements:
   ```
   pip install pandas openai duckdb python-dotenv openpyxl google-cloud-secret-manager
   ```
3. Place your data file in the `data/` directory or specify the path when running the script.

## Notes

- The pipeline is modular: use the notebook for exploration or the script for automation.
- Prompts can be easily swapped or edited for different analysis needs.
- Use the enhanced script to ensure all records are properly analyzed.
- For large datasets, consider using smaller batch sizes (100-250) to prevent API truncation issues.