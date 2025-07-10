# Review Analyzer

A tool for analyzing customer review comments using OpenAI's GPT models and DuckDB.

## Project Overview

This project provides a system for analyzing customer product review comments. It:

1. Extracts key themes from each comment
2. Determines sentiment for each theme
3. Calculates overall sentiment for each comment
4. Stores results in DuckDB
5. Exports analysis to Excel and CSV

## Features

- Processes large datasets in batches
- Uses OpenAI GPT-4.1-nano for analysis
- Supports both customer sentiment and product feedback analysis
- Exports results to combined Excel and CSV files
- Modular pipeline: use interactively (notebook) or as a script

## Files

- `main.ipynb` - Interactive notebook for step-by-step analysis and prototyping
- `sentiment_analyzer.py` - Production-ready script for batch analysis and export
- `analyze_comments.py` - Legacy implementation using JSON parsing
- `analyze_with_functions.py` - Legacy implementation using OpenAI function calling
- `product_prompt.txt` - Prompt for product-focused analysis
- `customer_sentiment_prompt.txt` - Prompt for customer sentiment analysis

## Pipeline Overview

The current pipeline (as implemented in `sentiment_analyzer.py` and `main.ipynb`) does the following:

1. **Import data** (CSV, Excel, or Parquet) into DuckDB
2. **Extract comments** to a DataFrame
3. **Run both customer sentiment and product feedback analysis** using OpenAI GPT
4. **Export results**:
   - Combined Excel file with two sheets (Customer Feedback, Product Feedback)
   - Separate CSV files for each analysis

## Usage

### Option 1: Interactive Analysis in Jupyter Notebook

- Open `main.ipynb` for an interactive, step-by-step workflow.
- This notebook demonstrates data import, DuckDB usage, prompt engineering, and batch sentiment analysis with OpenAI.
- Recommended for exploration, prototyping, and custom analysis.

### Option 2: Automated Batch Analysis via Python Script

- Use `sentiment_analyzer.py` for a production-ready, scriptable pipeline.
- This script:
  - Imports data into DuckDB
  - Extracts comments to a DataFrame
  - Runs both customer sentiment and product feedback analysis using OpenAI
  - Exports results to combined Excel and CSV files
- Run with:
  ```
  python sentiment_analyzer.py
  ```
- You can also import and call `run_sentiment_pipeline()` from your own scripts for custom workflows.

### Option 3: Legacy Scripts

- `analyze_comments.py` - Original implementation using JSON parsing
- `analyze_with_functions.py` - Enhanced implementation using OpenAI function calling

## Customization

You can customize the analysis by modifying the prompt files:
- `product_prompt.txt` - For product-focused analysis
- `customer_sentiment_prompt.txt` - For customer sentiment analysis

## Output Format

The analysis produces structured output with:
- Theme columns (Theme 1, Theme 2, Theme 3, ...)
- Sentiment columns (Sentiment 1, Sentiment 2, Sentiment 3, ...)
- Total_Sentiment column for overall sentiment

Results are stored in DuckDB tables and exported to Excel and CSV. The combined Excel file contains two sheets: one for product feedback, one for customer sentiment.

## Setup

1. Set up your OpenAI API key in a `.env` file:
   ```
   OPEN_API_KEY=your_api_key_here
   ```
   Or, if using Google Secret Manager, set the secret path:
   ```
   OPEN_API_KEY=projects/your_project_id/secrets/your_secret_name/versions/latest
   ```

2. Install requirements:
   ```
   pip install pandas openai duckdb python-dotenv openpyxl google-cloud-secret-manager
   ```

3. Place your data file (CSV, Excel, or Parquet) in the `data/` directory or specify the path when running the script.

## Performance Considerations

- Batch size can be adjusted based on your needs (default: 100)
- Larger batch sizes may be more efficient but use more memory
- Consider model choice based on budget and accuracy requirements

## Notes

- The pipeline is modular: you can use the notebook for exploration or the script for automation.
- Prompts can be easily swapped or edited for different analysis needs.
- The script supports both local API keys and GCP Secret Manager for authentication.

## Future Steps

- For full production readiness - defined data input and outputs are required with automation. Downstream analytics and dashboarding to utilize analytics. 