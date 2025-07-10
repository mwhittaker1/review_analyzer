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
- Uses OpenAI GPT-4.1-nano for analysis
- Supports both customer sentiment and product feedback analysis
- Exports results to combined Excel and CSV files
- Modular pipeline: use interactively (notebook) or as a script

## Key Files

- `main.ipynb` – Interactive notebook for step-by-step analysis and prototyping
- `sentiment_analyzer.py` – Script for batch analysis and export
- `prompts/` – Contains prompt templates for different analysis types

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

- Use `sentiment_analyzer.py` for a scriptable pipeline.
- Run with:
  ```
  python sentiment_analyzer.py
  ```
- You can also import and call `run_sentiment_pipeline()` from your own scripts.

## Customization

- Modify prompt files in `prompts/` to adjust analysis behavior.

## Output

- Theme columns (Theme 1, Theme 2, ...)
- Sentiment columns (Sentiment 1, Sentiment 2, ...)
- `Total_Sentiment` column for overall sentiment
- Results stored in DuckDB and exported to Excel/CSV

## Setup

1. Set your OpenAI API key in a `.env` file:
   ```
   OPEN_API_KEY=your_api_key_here
   ```
2. Install requirements:
   ```
   pip install pandas openai duckdb python-dotenv openpyxl
   ```
3. Place your data file in the `data/` directory or specify the path when running the script.

## Notes

- The pipeline is modular: use the notebook for exploration or the script for automation.
- Prompts can be easily swapped or edited for different analysis needs.