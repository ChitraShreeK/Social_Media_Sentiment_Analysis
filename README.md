## REDDIT SENTIMENT ANALYSIS: TECH COMMUNITY TREND FORECASTING

### Executive Summary
This project presents an interactive Streamlit dashboard that analyzes **111,000+ Reddit comments** from the r/technology subreddit. The goal is to quantify public opinion, emotional trends, and key conversational topics related to major technology themes like **AI, Privacy, and Political implications**.

The solution is driven by **advanced NLP techiques**, including **VADER (Valence Aware Dictionary and sEntiment Reasoner)** for sentiment scoring and the **NRC Word-Emotion Association Lexicon** for granular emotion detection.

### Business Problem
The tech industry thrives on **public trust** and adoption of innovations like AI and data driven systems. However, polarized debates around **privacy violations**, **political influence** and **ethical AI** fuel skepticism. For tech companies to proactively understand these **emotional undercurrents** the solution provides a reliable way to:
1. Detect shifts in **public sentiment** before they escalate
2. Understand **emotional responses** tied to technology topics
3. Align decisions with **consumer and community perspectives**

This project delivers the **data infrastructure and visualization tools** necessary to transform unstructured text into quantifiable, high value intelligence.

### Methodology
This analysis was executed through a robust NLP pipeline designed for speed and clarity. The data was passed through multi stage pipeline:
1. **Data Collection:** Scraped Reddit data using **Python** and **Python Reddit API Wrapper (PRAW)** from r/technology
2. **Cleaning & Standardization:** Raw comments were preprocessed using **tokenization**, **stopword removal** and **lemmatization** to ensure accurate feature extraction
3. **Sentiment Analysis:** The **VADER lexicon** was applied to assign a continuous **sentiment score** to each post, categorizing it as a **Positive**, **Negative** or **Neutral**
4. **Emotional Analysis:** The **NRC Word Emotion Association Lexicon** was leveraged to count the presence of words associated with **10 distinct emotions**. For eg: Trust, Anger, Anticipation etc within each comment
5. **Dashboard Visualization:** The analysis deployed via a Streamlit dashboard with Plotly gauges featuring three key views:
  - **Word Cloud:** Displays the **raw frequency** of keywords and **bigrams** to reveal prevailing discourse topics
  - **Emotion Analysis:** Presents **clustered horizontal bar charts** for all emotions within the selected topic
  - **Post Analysis:** A dynamic searchable data table for **qualitative drill down**, allowing users to filter posts by specific Topic, Sentiment and Emotion

### Skills Used
- **Language:** **Python** (Pandas for data wrangling, NLTK for text preprocessing), **SQL** (MySQL for EDA)
- **Visualization & Deployment:** Streamlit, Plotly, Matplotlib/Wordcloud
- **Data Science:** Natural language Processing (NLP), Sentiment Analysis, Text Mining, Lexicon-Based Modeling (VADER & NRC)

### Results
- Public discussion on **AI & Privacy** shows **high polarity**
- **Political topics** generate the most negative sentiment, signaling mistrust in governance
- Word clouds + emotion distribution charts reveal that **`companies`**, **`government`** and **`AI`** are top recurring drivers of sentiment
- **Negative sentiment (35.3%)** slightly outweighs **positive (33.5%)** highlighting a clear industry trust deficit

### Dashboard Preview

![Dashboard Main](dashboard\dashboard_images\01_dashboard.png)
*Word Cloud View*

---

![Dashboard Main](dashboard\dashboard_images\02_dashboard.png)
*Emotion Analysis View*

### Business Recommendation
- Adopt a communication strategy that emphasizes **user control, transparency and freedom** in product
- Product launches should focus on reliability, security and long term viability

### Next Steps & Limitations

#### Next Steps
1. Add time series analysis to track how sentiment changes after events
2. Integrate topic modeling to uncover hidden themes beyond keywords
3. Expand analysis to multiple subreddits for broader coverage

#### Limitations
1. VADER performs well on social media text but may miss sarcasm or nuanced context
2. Analysis is limited to English language posts
