CREATE DATABASE sentiment_analysis_db;

SHOW CREATE DATABASE sentiment_analysis_db;

USE sentiment_analysis_db;

CREATE TABLE post_and_comment(
	type VARCHAR(20),
    post_id VARCHAR(50),
    title TEXT,
    timestamp DATETIME,
    text TEXT,
    score INT,
    total_comments INT,
    post_url TEXT,
    text_to_analyze TEXT,
    cleaned_text TEXT
) CHARACTER SET utf8mb4;

-- Creating a new column to store the topic
ALTER TABLE post_and_comment
ADD COLUMN topic VARCHAR(50);
