import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter 

# loading the dataset
@st.cache_data
def load_data():
  data = pd.read_csv("../data/sentiment_analysis_dataset.csv")
  return data

df = load_data()

# writing function for tab 1: creating the word frequency data function to generate the 'word cloud'
stop_words = set(stopwords.words('english'))

custom_stopwords = {
  'peopl', 'like', 'get', 'dont', 'go', 'one', 'use', 'us', 'make', 'would', 'even',
  'want', 'time', 'think', 'im', 'right', 'need', 'thing', 'know', 'say', 'fuck',
  'year', 'that', 'also', 'way', 'work', 'tri', 'see', 'good', 'actual', 'still',
  'much', 'cant', 'realli', 'theyr', 'well', 'never', 'take', 'doesnt', 'could',
  'look', 'mean', 'happen', 'back', 'shit', 'car', 'point', 'said'
}

all_stopwords = stop_words.union(custom_stopwords)

# combingin all words
all_words = " ".join(df['cleaned_text'].astype(str)).split()
filtered_words = [w for w in all_words if w not in all_stopwords]
word_freq = Counter(filtered_words)

st.set_page_config(
  page_title = "Reddit Sentiment Analysis",
  layout = "wide"
)

st.title("Title of the Project")
st.markdown("Small deep down about the project")

########################################################

# Thisfuction will be used in tab2

# writing all the emotion column names
emotion_columns = ['positive', 'trust', 'joy', 'negative', 'anticipation', 'fear', 'surprise', 'anger', 'disgust', 'sadness']

# function to get the top words per emotion in topic wise
@st.cache_data
def get_top_emotion_words(df, topic, emotion_col, num = 5):
  """Finds the top words in 'cleaned_text' that triggered a specific emotion for a given topic"""

  # filter by Topic
  if topic != 'All':
    df_filtered = df[df['topic']  == topic].copy()
  else:
    df_filtered = df.copy()

  # filtering where the rows for emotions score is > 0
  df_emotion = df_filtered[df_filtered[emotion_col] > 0].copy()

  if df_emotion.empty:
    return {}
  
  # aggregating all the words
  all_text = " ".join(df_emotion['cleaned_text'].astype(str)).lower()
  words = all_text.split()
  
  # recalculating word frequency by excluding all the stop words to get the most frequent words for that emotion
  word_freq_emotion = Counter([w for w in words if w not in all_stopwords and len(w) > 1])

  return dict(word_freq_emotion.most_common(num))

# function to get emotion count
def get_emotion_distribution(df, topic):
  """Calculates the sum of emotion scores and converts them to normalized percentage"""
  if topic != 'All':
    df_topic = df[df['topic'] == topic].copy()
  else:
    df_topic = df.copy()

  # calculating the total sum of emotion linked words
  emotion_counts = df_topic[emotion_columns].sum().reset_index()
  emotion_counts.columns = ['Emotion', 'Count']

  # calculating the percentage
  total_words = emotion_counts['Count'].sum()
  if total_words > 0:
    emotion_counts['Percentage'] = (emotion_counts['Count'] / total_words) * 100
  else:
    emotion_counts['Percentage'] = 0
  
  emotion_counts = emotion_counts[emotion_counts['Percentage'] > 0]

  return emotion_counts.sort_values(by = 'Percentage', ascending = False)

##################################################################

# displaying key metrics of positive, negative and neutral score

# creating new column to categorize sentiment
def categorize_sentiment(score):
  if score >= 0.05:
    return "Positive"
  if score <= -0.05:
    return "Negative"
  else:
    return "Neutral"
  
df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)

# calculating the average scores for each category
pos_avg = df[df['sentiment_category'] == 'Positive']['sentiment_score'].mean()
neg_avg = df[df['sentiment_category'] == 'Negative']['sentiment_score'].mean()
neu_avg = df[df['sentiment_category'] == 'Neutral']['sentiment_score'].mean()

# calculating the total count for each category
pos_count = df[df['sentiment_category'] == 'Positive'].shape[0]
neg_count = df[df['sentiment_category'] == 'Negative'].shape[0]
neu_count = df[df['sentiment_category'] == 'Neutral'].shape[0]

# dispalying the key metrics of average positive, negative and neutral
col1, col2, col3 = st.columns(3, border = True)

with col1:
  st.metric(label = "Average Positive Score", value = f"{pos_avg:.2f}")
  st.markdown(f"**Total Positive Posts:** **{pos_count:,}**")

with col2:
  st.metric(label = "Average Negative Score", value = f"{neg_avg:.2f}")
  st.markdown(f"**Total Negative Posts:** **{neg_count:,}**")

with col3:
  st.metric(label = "Average Neutral Score", value = f"{neu_avg:.2f}")
  st.markdown(f"**Total Neutral Posts:** **{neu_count:,}**")

# creating tabs to explore sentiment analysis
tab1, tab2, tab3 = st.tabs([
                            "Word Cloud",
                            "Emotion Analysis",
                            "Post Analysis"
])

###############################
# TAB 1: displaying the top most used words
with tab1:
  # adding columns on the left side the silder will be displayed to choose how many words to display and on the right right selected number of most used words will be displayed
  select_col, word_display_col = st.columns([1.5, 3])

  # Adding slider to choose the number of words to display
  with select_col:
    st.subheader("")
    top_words_selection = st.select_slider(
      "**Select the number of Top Words:**",
      options = [
        "10",
        "15",
        "20",
        "25",
        "30",
        "50",
        "100"
      ],
      value = "30"
    )

    # converting the selected string to an interger
    n_num = int(top_words_selection)

    # adding a checkbox to switch between single words or two phrased words
    use_bigrams = st.checkbox("Show Two Word Phrases", value = False)

    st.subheader("Raw Frequency Data")
    st.caption("Shows the exact count for the selected words/pharses")

    if use_bigrams:
      bigrams = ngrams(filtered_words, 2)
      bigrams_freq = Counter(bigrams)
      bigram_dict = {"_".join(k): v for k, v in bigrams_freq.most_common(n_num)}
      st.dataframe(pd.Series(bigram_dict).head(n_num), use_container_width = True)
    else:
      st.dataframe(pd.Series(word_freq).head(n_num), use_container_width = True)

  with word_display_col:
    st.subheader(f"Top {n_num} Most Used Words")

    # including bigrams to calculate word frequency
    if use_bigrams:
      # same code used in EDA
      bigrams = ngrams(filtered_words, 2)
      bigrams_freq = Counter(bigrams)

      # combining the bigram words into a single string
      bigram_dict = {"_".join(k): v for k, v in bigrams_freq.most_common(n_num)}
      wordcloud = WordCloud(width = 800, height = 400, background_color = "white").generate_from_frequencies(bigram_dict)
    else:
      wordcloud = WordCloud(width = 800, height = 400, background_color = "white").generate_from_frequencies(dict(word_freq.most_common(n_num)))

    # displaying the Word Cloud
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.imshow(wordcloud, interpolation = "bilinear")
    ax.axis("off")
    st.pyplot(fig)

###############################################
# TAB 2: In this tab, using bar chart to display the emotions based on the AI, Privacy, Political related and overall emotions
with tab2:
  st.subheader("Emotional Landscape of r/technology Subreddit Discussions")
  
  # adding checkbox
  topic_selection = st.radio(
    "**Select Topic for Emotion Analysis:**",
    options = [
      "All",
      "AI Related",
      "Privacy Related",
      "Political Related",
      "AI and Privacy Related",
      "Privacy and Political Related"
    ],
    index = 0,
    horizontal = True
  )

  st.header(f"Emotion Distribution for: {topic_selection}")

  # emotion distribution chart
  emotion_df_plot = get_emotion_distribution(df, topic_selection)

  if not emotion_df_plot.empty:
    fig_emotion = px.bar(
      emotion_df_plot,
      x = 'Percentage',
      y = 'Emotion',
      orientation = 'h',
      text = 'Percentage',
      color = 'Emotion',
      color_discrete_sequence = px.colors.qualitative.Bold,
      title = f"Percentage of Emotional Words in '{topic_selection}' posts"
    )

    fig_emotion.update_traces(texttemplate = '%{text:.1f}%', textposition = 'outside')
    fig_emotion.update_layout(uniformtext_minsize = 8, uniformtext_mode = 'hide', yaxis = {'categoryorder':'total ascending'})
    st.plotly_chart(fig_emotion, use_container_width = True)

    st.markdown("---")
    st.header("Top Contextual Words Driving Key Emotions")

    # top 4 most prominent emotions in the selected topic for display
    top_emotions = emotion_df_plot['Emotion'].head(4).tolist()
    emotion_cols = st.columns(len(top_emotions)) # this will display the top most used words
    
    for i, emotion in enumerate(top_emotions):
      top_words_dict = get_top_emotion_words(df, topic_selection, emotion, num = 7)

      with emotion_cols[i]:
        st.subheader(f"{emotion.title()}")

        if top_words_dict:
          word_list_markdown = ""
          for word, count in top_words_dict.items():
            word_list_markdown += f"- **{word.upper()}** ({count:,})\n"
          st.markdown(word_list_markdown)
        else:
          st.markdown("*No significant words found for this emotion.*")
  else:
    st.info(f"No emotional words found for the topic: {topic_selection}")


#########################################
# TAB 3: this tab will allow to check specific topic and its sentiment and emotion from the comments
# Eg: slecting 'Negative' comments related to 'Political' that express 'Anger'
with tab3:
  st.subheader("Filter and view individual 'r/technology' subreddit comments to understand the context of the sentiment")

  filter_col1, filter_col2, filter_col3 = st.columns(3)

  with filter_col1:
    topic_filter = st.selectbox(
      "**Filter by Topic**",
      options = ['All Topics'] + df['topic'].unique().tolist(),
      index = 0
    )

  with filter_col2:
    sentiment_filter = st.selectbox(
      "**Filter by Sentiment**",
      options = ['All Sentiments'] + df['sentiment_category'].unique().tolist(),
      index = 0
    )

  with filter_col3:
      emotion_options = ['All Emotions'] + [e.title() for e in emotion_columns] + ['No Emotion']
      emotion_filter = st.selectbox(
        "**Filter by Specific Emotion**",
        options = emotion_options,
        index = 0
      )
  
  st.markdown("---")

  df_filtered = df.copy()

  # filter condition for topic
  if topic_filter != 'All Topics':
    df_filtered = df_filtered[df_filtered['topic'] == topic_filter]

  # filter condition for sentiment
  if sentiment_filter != 'All Sentiments':
    df_filtered = df_filtered[df_filtered['sentiment_category'] == sentiment_filter]

  # filter condition for emotion
  if emotion_filter != 'All Emotions':
    if emotion_filter == 'No Emotion':
      df_filtered = df_filtered[df_filtered[emotion_columns].sum(axis = 1) == 0]
    else:
      # this is for the specific emotion column > 0
      emotion_col_name = emotion_filter.lower()
      df_filtered = df_filtered[df_filtered[emotion_col_name] > 0]
    
  # Selecting only the relevant columns to display
  columns_to_show = [
    'cleaned_text',
    'topic',
    'sentiment_category',
    'sentiment_score'
  ]

  st.info(f"Displaying **{df_filtered.shape[0]:,}** posts matching your criteria")

  if not df_filtered.empty:
    st.dataframe(
      df_filtered[columns_to_show].rename(
        columns = {'cleaned_text': 'Comment Text',
                   'sentiment_category': 'Overall Sentiment',
                   'sentiment_score': 'VADER Score'}
      ),
      use_container_width = True,
      height = 600
    )
  else:
    st.warning("No posts found for the selected combination of filters")