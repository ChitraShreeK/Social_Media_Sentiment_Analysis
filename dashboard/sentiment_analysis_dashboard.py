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

try:
  nltk.data.find('corpora/stopwords')
except ImportError:
  try:
    nltk.data.find('corpora/stopwords')
  except LookupError:
    nltk.download('stopwords')

# loading the dataset
@st.cache_data
def load_data():
  data = pd.read_csv("data/sentiment_analysis_dataset.csv")
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

################################################

# page configuration
st.set_page_config(
  page_title = "Reddit Sentiment Analysis",
  layout = "wide",
  initial_sidebar_state = "collapsed"
)

# Project tilte
logo_col, title_col = st.columns([0.1, 0.9])
with logo_col:
  st.image("../dashboard/images/Reddit_Icon_FullColor.svg", width = 60)

with title_col:
  st.markdown("<div style='display: flex; align-items: center; height: 60px;'><h1 style = 'margin: 0; padding: 0;'>Reddit Sentiment Analysis: The Tech Community's Voice</h1></div>", unsafe_allow_html=True)

st.markdown("""
      **Analyzing public sentiment on r/technology** to uncover emotional trends and key topics in discussions 
      about **AI, Privacy, and Political** issues. This dashboard quantifies complex text into actionable insights.
      """)

st.markdown("---")

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

# Calculating the percentage to show the difference between category's average score and neutral score
pos_dev_percent = (pos_avg / 1.0) * 100
neg_dev_percent = (abs(neg_avg) / 1.0) * 100
neu_dev_percent = (neu_avg / 1.0) * 100


# creating micro chart function for the metrics visuals
def create_percentage_gauge(value, color, max_value):
  percent = (value / max_value) * 100

  fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = percent,
    number = {
      'valueformat': '.1f',
      'suffix': "%",
      'font': {'size': 24, 'color': color}
    },
    gauge = {
      'axis': {'range': [None, 100], 'tickwidth': 0, 'visible': False},
      'bar': {'color': color, 'thickness': 0.8},
      'bgcolor': "lightgray",
      'borderwidth': 0,
      'steps': [
        {'range': [0, 100], 'color': 'lightgray'}
      ],
      'threshold': {
        'line': {'color': color, 'width': 4},
        'thickness': 0.75,
        'value': percent
      }
    }
  ))

  fig.update_layout(
    height = 150,
    margin = dict(l = 10, r = 10, t = 10, b = 10)
  )
  return fig


# dispalying the key metrics of average positive, negative and neutral
total_posts = df.shape[0]
col1, col2, col3 = st.columns(3, border = True)

with col1:
  text_col, chart_col = st.columns([1.5, 1])

  with text_col:
    st.metric(
      label = "Average Positive Score",
      value = f"{pos_avg:.2f}",
      delta = f"{pos_dev_percent:.1f}% Deviation",
      delta_color="normal"
    )
    st.markdown(f"**Total Positive Posts:** **{pos_count:,}**")
  
  with chart_col:
    #st.caption(f"{round((pos_count / total_posts) * 100, 1)}% of all posts")
    gauge_pos = create_percentage_gauge(pos_count, '#4CAF50', total_posts)
    st.plotly_chart(gauge_pos, use_container_width = True, config = {'displayModeBar': False})

with col2:
  text_col, chart_col = st.columns([1.5, 1])

  with text_col:
    st.metric(
      label = "Average Negative Score",
      value = f"{neg_avg:.2f}",
      delta = f"{neg_dev_percent * -1:.1f}% Deviation",
      delta_color = "normal"
    )
    st.markdown(f"**Total Negative Posts:** **{neg_count:,}**")

  with chart_col:
    #st.caption(f"{round((neg_count / total_posts) * 100, 1)}% of all posts")
    gauge_neg = create_percentage_gauge(neg_count, '#F44336', total_posts)
    st.plotly_chart(gauge_neg, use_container_width = True, config = {'displayModeBar': False})

with col3:
  text_col, chart_col = st.columns([1.5, 1])

  with text_col:
    st.metric(
      label = "Average Neutral Score",
      value = f"{neu_avg:.2f}",
      delta = f"{neu_dev_percent:.1f}% Deviation",
      delta_color = "off"
    )
    st.markdown(f"**Total Neutral Posts:** **{neu_count:,}**")

  with chart_col:
    #st.caption(f"{round((neu_count / total_posts) * 100, 1)}% of all posts")
    gauge_neu = create_percentage_gauge(neu_count, '#FFC107', total_posts)
    st.plotly_chart(gauge_neu, use_container_width = True, config = {'displayModeBar': False})


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
    st.subheader("Configuration")
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
      value = "25"
    )

    # converting the selected string to an interger
    n_num = int(top_words_selection)

    # adding a checkbox to switch between single words or two phrased words
    use_bigrams = st.checkbox("Show Two Word Phrases", value = False)

    # data calculation
    if use_bigrams:
      bigrams = ngrams(filtered_words, 2)
      bigrams_freq = Counter(bigrams)
      final_word_dict = {"_".join(k): v for k, v in bigrams_freq.most_common(n_num)}
    else:
      final_word_dict = dict(word_freq.most_common(n_num))

  # display the raw frequency expander
  with select_col:
    st.markdown("---")
    with st.expander("View Raw Frequency Table"):
      st.caption("Shows the exact count for the top {n_num} selected words/phrases")
      st.dataframe(pd.Series(final_word_dict), use_container_width = True)

  # Word Cloud Display
  with word_display_col:
    st.subheader(f"Word Cloud of Top {n_num} Words")
    wordcloud = WordCloud(width = 800, height = 500, background_color = "white").generate_from_frequencies(final_word_dict)

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

  st.markdown("---")
  st.header(f"Emotion Distribution for: {topic_selection}")
  st.caption("Each bar chart shows the most frequent words that triggered a specific emotion in the posts")

  # emotion distribution chart
  emotion_df_plot = get_emotion_distribution(df, topic_selection)

  if not emotion_df_plot.empty:
    emotions_to_display = emotion_columns

    columns_per_row = 4

    for i in range(0, len(emotions_to_display), columns_per_row):
      emotion_cols = st.columns(columns_per_row)

      # iterating through the emotions
      for j, emotion in enumerate(emotions_to_display[i:i + columns_per_row]):
        with emotion_cols[j]:
          top_words_dict = get_top_emotion_words(df, topic_selection, emotion, num = 7)

          #customizing the colors
          if emotion in ['negative', 'anger', 'disgust', 'fear', 'sadness']:
            bar_color = '#F44336'
          elif emotion in ['positive', 'joy', 'trust', 'anticipation', 'surprise']:
            bar_color = '#4CAF50'
          else:
            bar_color = '#FFC107'
          
          if top_words_dict:
            df_words = pd.DataFrame(
              list(top_words_dict.items()),
              columns = ['Word', 'Count']
            )

            fig_word_bar = px.bar(
              df_words.sort_values('Count', ascending = True),
              x = 'Count',
              y = 'Word',
              orientation = 'h',
              text = 'Count',
              color_discrete_sequence = [bar_color]
            )

            fig_word_bar.update_traces(
              marker_line_width = 0,
              texttemplate = '%{text:,}',
              textposition = 'outside'
            )

            fig_word_bar.update_layout(
              title = f"{emotion.title()}",
              title_font_size = 16,
              height = 350,
              margin = dict(l = 10, r = 40, t = 50, b = 10),
              plot_bgcolor = 'rgba(240, 240, 240, 0.1)',
              paper_bgcolor = 'rgba(0, 0, 0, 0)',
              # xaxis_title = None,
              # yaxis_title = None,
              xaxis = dict(
                showgrid = True,
                showline = True,
                linecolor = '#888888',
                ticks = 'outside',
                ticklen = 5,
                range = [0, df_words['Count'].max() * 1.2]
              ),
              yaxis = dict(
                showgrid = False,
                showline = False
              )
            )

            st.plotly_chart(fig_word_bar, use_container_width = True, config = {'displayModeBar': False})
          
          else:
            st.markdown(f"**{emotion.title()}**")
            st.markdown(f"**No words found for {emotion}**")
  else:
    st.info(f"No emotional words found for the topic: {topic_selection}")


#########################################
# TAB 3: this tab will allow to check specific topic and its sentiment and emotion from the comments
# Eg: slecting 'Negative' comments related to 'Political' that express 'Anger'
with tab3:
  st.subheader("Filter and view individual 'r/technology' subreddit comments to understand the context of the sentiment")

  sentiment_map = {
    'Positive': '😀 Positive',
    'Negative': '😡 Negative',
    'Neutral': '😐 Neutral'
  }

  emotion_display_map = {
    'positive': '🟢 Positive',
    'negative': '🔴 Negative',
    'trust': '🤝 Trust',
    'fear': '😨 Fear',
    'joy': '😊 Joy',
    'anger': '😡 Anger',
    'surprise': '😲 Surprise',
    'disgust': '🤢 Disgust',
    'sadness': '😢 Sadness',
    'anticipation': '⏳ Anticipation'
  }

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

  df_display = df_filtered.copy()

  if not df_display.empty:
    df_display['sentiment_category'] = df_display['sentiment_category'].map(sentiment_map)

    if emotion_filter != 'All Emotions':
      if emotion_filter != 'No Emotion':
        display_emotion = emotion_filter.lower()
        df_display['filtered_emotion'] = df_display[display_emotion].apply(
          lambda x: emotion_display_map[display_emotion] if x > 0 else '-'
        )

        # Selecting only the relevant columns to display
        columns_to_show = [
          'cleaned_text',
          'topic',
          'sentiment_category',
          'filtered_emotion',
          'sentiment_score'
        ]
      else:
        columns_to_show = [
          'cleaned_text',
          'topic',
          'sentiment_category',
          'snetiment_score'
        ]
    else:
      columns_to_show = [
        'cleaned_text',
        'topic',
        'sentiment_category',
        'sentiment_score'
      ]
  else:
    columns_to_show = []
  

  st.info(f"Displaying **{df_filtered.shape[0]:,}** posts matching your criteria")

  if not df_filtered.empty:
    st.dataframe(
      df_display[columns_to_show].rename(
        columns = {'cleaned_text': 'Comment Text',
                   'sentiment_category': 'Overall Sentiment',
                   'sentiment_score': 'VADER Score',
                   'sentiment_score': 'VADER Score'}
      ),
      use_container_width = True,
      height = 600
    )
  else:

    st.warning("No posts found for the selected combination of filters")




