SELECT * FROM post_and_comment LIMIT 10;

SELECT COUNT(*) FROM post_and_comment;  -- total count is 14053

-- checking for Null values
SELECT
  SUM(
    CASE WHEN cleaned_text IS NULL OR cleaned_text = '' THEN 1 ELSE 0 END
  ) AS missing_cleaned_text,
  SUM(
    CASE WHEN text_to_analyze IS NULL OR text_to_analyze = '' THEN 1 ELSE 0 END
  ) AS missing_text_to_analyze
FROM post_and_comment;
-- cleaned_text column is from the 'text_to_analyze' so finding 154 columns are null so checking which rows are actually null is it due to 'stopwords' OR 'emojis' or 'url'

SELECT
  text_to_analyze,
  cleaned_text
FROM post_and_comment
WHERE cleaned_text IS NULL OR cleaned_text = '' OR cleaned_text = ' ';
-- By the output its very clear that these 154 rows are null due to stopwords, mentions, hashtags, urls

-- Output confirmed that null values are due to text cleaning so filling these null vales with empty string to ensure the data consistency
UPDATE post_and_comment
SET cleaned_text = ''
WHERE cleaned_text IS NULL OR TRIM(cleaned_text) = '';


SELECT
  SUM(  
    CASE WHEN cleaned_text IS NULL THEN 1 ELSE 0 END
  ) AS null_count,
  SUM(
    CASE WHEN cleaned_text = '' THEN 1 ELSE 0 END
  ) AS empty_string_count
FROM post_and_comment; -- The output confirmed that 'null' values are replaced with empty string

-- rechecking all the coulmns for Null values
SELECT
  COUNT(*) AS total_rows,  -- total_rows confirms 14053
  SUM(CASE WHEN type IS NULL THEN 1 ELSE 0 END) AS type_nulls,
  SUM(CASE WHEN post_id IS NULL THEN 1 ELSE 0 END) AS post_id_nulls,
  SUM(CASE WHEN title IS NULL THEN 1 ELSE 0 END) AS title_nulls,
  SUM(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END) AS timestamp_nulls,
  SUM(CASE WHEN text IS NULL THEN 1 ELSE 0 END) AS text_nulls,
  SUM(CASE WHEN score IS NULL THEN 1 ELSE 0 END) AS score_nulls,
  SUM(CASE WHEN total_comments IS NULL THEN 1 ELSE 0 END) AS total_comments_nulls,
  SUM(CASE WHEN post_url IS NULL THEN 1 ELSE 0 END) AS post_url_nulls,
  SUM(CASE WHEN text_to_analyze IS NULL THEN 1 ELSE 0 END) AS text_to_analyze_nulls,
  SUM(CASE WHEN cleaned_text IS NULL THEN 1 ELSE 0 END) AS cleaned_text_nulls
FROM post_and_comment;  -- all rows have 0 null count

-- starting with the deeper EDA

-- Average score for posts and comments
SELECT
  type,
  AVG(score) AS average_score
FROM post_and_comment
GROUP BY type;  -- the avg score for posts is 3189.5900 and for comments its 37.8276

-- checking for the count
SELECT
  type,
  COUNT(*) AS type_count
FROM post_and_comment
GROUP BY type;   -- from the r/technology subreddit extracted 100 posts so the output confirms that post count is 100 and comments count is 13953

-- checking for the average number of comments per post
SELECT
  AVG(total_comments) AS avg_comment_per_post
FROM post_and_comment
WHERE type = 'Post';  -- the avg comments per post is 224.7500

-- looking for any outliers by score distribution
SELECT
  MIN(score) AS min_score,    -- min_score: -215
  MAX(score) AS max_score,    -- max_score: 115279
  AVG(score) AS avg_score     -- avg_score: 60.2552
FROM post_and_comment;

SELECT * FROM post_and_comment LIMIT 10;

-- Identifying top 10 posts based on the score
SELECT
  type, post_id, title, score, total_comments
FROM post_and_comment
WHERE type = 'Post'
ORDER BY score DESC
LIMIT 10;   -- got the top 10 posts based on the score, first is with the post_id: 1nimpmu with the score: 115279 and total_comments: 2663 and the 10th highest is from post_id:1njl1e8 with score: 6066 and total_comments: 947

-- Identifying top 10 comments based on the score
SELECT
  type, text, post_id, score
FROM post_and_comment
WHERE type = 'comment'
ORDER BY score DESC
LIMIT 10;   -- got the top 10 comments based on the score, first highest comment from post_id:1nimpmu with score of 24821 and the 10th highest comment from the post_id:1njnb5k with score: 6620

-- Time Based Exploration

-- checking the number of posts and comments per day
SELECT
  DATE(timestamp) AS date,
  COUNT(*) AS post_count
FROM post_and_comment
GROUP BY DATE(timestamp)
ORDER BY date;  -- Output: for 16-09-2025 the post_count: 3028, 17-09-2025 post_count: 6517, 18-09-2025 post_count:4508

-- Reason for going only for date: Scraped hot 100 posts from the r/technology subreddit as reddit is one of the current popular social media there is a chance of higher posts and comments so for now didn't go further with month

-- looking for peak post hours
SELECT
  HOUR(timestamp) AS hours,
  COUNT(*) AS post_count
FROM post_and_comment
GROUP BY HOUR(timestamp)
ORDER BY post_count DESC; 

-- looking at the output its clear that peak posting hours is during lunch time. At 13hrs(1 PM) post_count: 1005, 14hrs(2 PM) post_count:962, 12hrs(12 PM) post_count: 860, 20hrs(8PM) post_count: 915
-- less posting hours are 8hrs(8 AM) post_count: 227, 7hrs(7 AM) post_count:250, 6hrs(6 AM) post_count:250


-- Average length of posts and comments
SELECT
  type,
  AVG(CHAR_LENGTH(text)) AS avg_text_length,
  AVG(CHAR_LENGTH(cleaned_text)) AS avg_cleaned_text_length
FROM post_and_comment
GROUP BY type;
-- posts dont have much text so column length ~1, cleaned length = 0, whereas comments have alot of text so the length ~162, avg cleaned length ~89

-- Moving to 03_eda.ipynb file to find out the common words used in the scrapped dataset. Using SQL its difficult to get the output so switching to Pandas for this.
