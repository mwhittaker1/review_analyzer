DECLARE
  PROMPT_TEMPLATE STRING DEFAULT """
You are an expert theme extractor for customer surveys. Your job is to read through customer comments and pull out all the key themes in each one. The themes should capture the main subjects/topics mentioned in the comment with key insights. Avoid overly generic descriptions such as customer dissatisfaction. Instead describe the causal themes. Themes should be simple, succinct and repeatable. Assign these themes a sentiment of positive, negative, or neutral. The number of themes extracted from a single review should be at least one but no more than the number of independent clauses in the survey. For a single survey comment, the themes should not be repetitive. Each theme should be at least two words but no more than 4. Extract themes in the format:

Theme: Count: Sentiment: Comment_ids: END 

Theme: Count: Sentiment: Comment_ids: END

Where Theme is followed by the brief synopsis, count is followed by the number of comments with that theme, sentiment followed by is one of positive, negative or neutral, and comment_ids is followed by the list of comment_ids that had the associated theme.  

For example:

Theme: Quick shipping Count: 4 Sentiment: Positive Comment_ids: 45, 92, 31, 89

    REVIEWS FOLLOW:
    ---
""";
DECLARE
  SUMMARY_PROMPT_TEMPLATE STRING DEFAULT """

Read through the customer survey comment themes and extract all the key themes in each one in short and simple and repeatable phrases that will enable us to see if these themes are trending. Do not return new themes or change the sentiment they were assigned, only simplify the theme descriptions when appropriate. If a theme has more than four words or a / then simplify it. Only return the list of key themes and the count of how many times they occur and the sentiment assigned. Do not return any other text nor an explanation of what you did. Each theme should be at least two words but no more than 4. Return only themes, the counts of those themes, the assigned sentiment, and the list of assigned comment_ids in the following format:

Theme: Count: Sentiment: Comment_ids: END 

Theme: Count: Sentiment: Comment_ids: END

Where Theme is followed by the brief synopsis, count is followed by the number of comments with that theme, sentiment followed by is one of positive, negative or neutral, and comment_ids is followed by the list of comment_ids that had the associated theme.  


""";
--Check if the table already exists
IF (SELECT size_bytes FROM urbn-ds-pipelines-dev.customer_surveys.__TABLES__ WHERE table_id='an_survey_themes') > 0
----------------------------------
-- IF TABLE EXISTS, INSERT INTO IT
----------------------------------
THEN
insert into urbn-ds-pipelines-dev.customer_surveys.an_survey_themes
with chunked_reviews AS (
  SELECT
    Survey,
    Topic,
    chunk_id, 
    Comment_date,  
    STRING_AGG(numbered_comment, '\n ') AS chunked_review
  FROM
    urbn-ds-pipelines-dev.customer_surveys.an_chunked_reviews 
  WHERE Comment_date > (SELECT max(Comment_date) from urbn-ds-pipelines-dev.customer_surveys.an_survey_themes)
  GROUP BY
    Survey, Topic, Comment_date, chunk_id
),
initial_summary as (
SELECT  Survey,
        Topic,   
        Comment_date,
        chunked_review,
        ml_generate_text_llm_result AS Summary,
    
  FROM
    ML.GENERATE_TEXT(MODEL `urbn-ds-pipelines-dev.customer_surveys.claude-3-7`,
      (
        SELECT
        Survey,
        Topic, 
        Comment_date,
        chunk_id,
        chunked_review,
        CONCAT(PROMPT_TEMPLATE, chunked_review) AS prompt
      FROM
        chunked_reviews
    AS prompt
      ),
      STRUCT(TRUE AS flatten_json_output)) 
)
SELECT Survey,
       Topic,
       Comment_date,
       chunked_review,
       ml_generate_text_llm_result AS Summary,  
  FROM
      ML.GENERATE_TEXT(MODEL`urbn-ds-pipelines-dev.customer_surveys.claude-3-7`,
        (
        SELECT
          Survey,
          Topic, 
          Comment_date,
          chunked_review,
          CONCAT(SUMMARY_PROMPT_TEMPLATE, Summary) AS prompt
        FROM
          initial_summary ),
        STRUCT(TRUE AS flatten_json_output))
;
----------------------------------
-- IF IT DOES NOT EXIST, CREATE IT
----------------------------------
ELSE
CREATE TABLE urbn-ds-pipelines-dev.customer_surveys.an_survey_themes
Partition by Comment_date AS
with chunked_reviews AS (
  SELECT
    Survey,
    Topic,
    chunk_id, 
    Comment_date,  
    STRING_AGG(numbered_comment, '\n ') AS chunked_review
  FROM
    urbn-ds-pipelines-dev.customer_surveys.an_chunked_reviews 
  GROUP BY
    Survey, Topic, Comment_date, chunk_id
),
initial_summary as (
SELECT  Survey,
        Topic,   
        Comment_date,
        chunked_review,
        ml_generate_text_llm_result AS Summary,
    
  FROM
    ML.GENERATE_TEXT(MODEL `urbn-ds-pipelines-dev.customer_surveys.claude-3-7`,
      (
        SELECT
        Survey,
        Topic, 
        Comment_date,
        chunk_id,
        chunked_review,
        CONCAT(PROMPT_TEMPLATE, chunked_review) AS prompt
      FROM
        chunked_reviews
    AS prompt
      ),
      STRUCT(TRUE AS flatten_json_output)) 
)
SELECT Survey,
       Topic,
       Comment_date,
       chunked_review,
       ml_generate_text_llm_result AS Summary,  
  FROM
      ML.GENERATE_TEXT(MODEL`urbn-ds-pipelines-dev.customer_surveys.claude-3-7`,
        (
        SELECT
          Survey,
          Topic, 
          Comment_date,
          chunked_review,
          CONCAT(SUMMARY_PROMPT_TEMPLATE, Summary) AS prompt
        FROM
          initial_summary ),
        STRUCT(TRUE AS flatten_json_output))
;

END IF;
