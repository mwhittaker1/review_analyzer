# Review Analyzer

A tool for analyzing customer review comments using OpenAI's GPT models.

## Project Overview

This project provides a system for analyzing customer product review comments. It:

1. Extracts key themes from each comment
2. Determines sentiment for each theme
3. Calculates overall sentiment for each comment
4. Stores results in DuckDB
5. Exports analysis to Excel

## Features

- Processes large datasets in batches
- Uses OpenAI GPT-4.1-nano for analysis
- Provides multiple implementation options (regular JSON parsing vs. function calling)
- Exports results to Excel for further analysis

## Files

- `analyze_comments.py` - Original implementation using JSON parsing
- `product_prompt.txt` - Prompt for product-focused analysis
- `customer_sentiment_prompt.txt` - Prompt for customer sentiment analysis
- `function_prompt.txt` - Prompt optimized for function calling

## Implementation Approaches

### 1. JSON Parsing (Original)

The original implementation sends comments to the OpenAI API and parses JSON responses, with fallback to custom pattern matching.

**Pros:**
- Works with a variety of response formats
- Has fallback parsing for non-JSON responses

**Cons:**
- Less reliable structure in responses
- Requires more post-processing
- Occasionally needs error handling for malformed responses

## Usage

1. Set up your OpenAI API key in a `.env` file:
   ```
   OPEN_API_KEY=your_api_key_here
   ```

2. Install requirements:
   ```
   pip install pandas openai duckdb python-dotenv openpyxl
   ```

3. Run the analyzer:
   ```
   python analyze_comments.py  # Original implementation
   ```
   
   Or the enhanced version:
   ```
   python analyze_with_functions.py  # Function calling implementation
   ```


## Customization

You can customize the analysis by modifying the prompt files:
- `product_prompt.txt` - For product-focused analysis
- `customer_sentiment_prompt.txt` - For customer sentiment analysis
- `function_prompt.txt` - For function calling implementation

## Output Format

The analysis produces a structured output with:
- Theme columns (Theme 1, Theme 2, Theme 3)
- Sentiment columns (Sentiment 1, Sentiment 2, Sentiment 3)
- Total_Sentiment column for overall sentiment

Results are stored in DuckDB tables and exported to Excel.

## Performance Considerations

- Batch size can be adjusted based on your needs (default: 50)
- Larger batch sizes may be more efficient but use more memory
- Consider model choice based on budget and accuracy requirements
