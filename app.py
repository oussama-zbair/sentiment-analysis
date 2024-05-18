from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from datetime import datetime
import requests
from mtranslate import translate
import string
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

app = Flask(__name__)

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.netloc == "www.hespress.com"
    except ValueError:
        return False

def arabic_to_english_month(arabic_month):
    months_arabic = ['يناير', 'فبراير', 'مارس', 'أبريل', 'ماي', 'يونيو', 'يوليوز', 'غشت', 'شتنبر', 'أكتوبر', 'نونبر',
                     'دجنبر']
    months_english = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                      'November', 'December']
    return months_english[months_arabic.index(arabic_month)]

def fetch_comments(url):
    user_name = []
    comment_text = []
    comment_date = []

    headers = {'User-Agent': 'Mozilla/5.0'}
    data = requests.get(url, headers=headers)
    soup = BeautifulSoup(data.content, "html.parser")

    article_title_tag = soup.find('h1', class_='post-title')
    article_title = article_title_tag.get_text() if article_title_tag else 'Unknown Title'

    c = soup.find('ul', {"class": "comment-list hide-comments"})

    if c:
        for li in c.find_all('li', class_='comment'):
            user_span = li.find('span', class_='fn heey')
            user_name.append(user_span.get_text() if user_span else 'Unknown')

            comment_p = li.find('p')
            comment_text.append(comment_p.get_text() if comment_p else 'No comment text found')

            date_span = li.find('div', class_='comment-date')
            if date_span:
                date_string = date_span.get_text().strip()
                date_parts = date_string.split()
                day = int(date_parts[1])
                month_arabic = date_parts[2]
                month = arabic_to_english_month(month_arabic)
                year = int(date_parts[3])
                time_parts = date_parts[-1].split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                comment_date.append(pd.Timestamp(year, datetime.strptime(month, '%B').month, day, hour, minute))
            else:
                comment_date.append('Unknown date')
    else:
        print("No comments found on the page.")

    df = pd.DataFrame({'User Name': user_name, 'Comment': comment_text, 'Date': comment_date})
    return df, article_title

def analyze_sentiment(comment):
    sid = SentimentIntensityAnalyzer()
    arabic_stopwords = set(stopwords.words('arabic'))
    arabic_punctuation = set(string.punctuation)

    try:
        translated_comment = translate(comment, 'en')
        tokens = translated_comment.split()
        filtered_tokens = [word for word in tokens if
                           word.lower() not in arabic_stopwords and word not in arabic_punctuation]
        preprocessed_comment = ' '.join(filtered_tokens)
        scores = sid.polarity_scores(preprocessed_comment)
        compound_score = scores['compound']
    except Exception as e:
        print("Error occurred during sentiment analysis:", e)
        compound_score = None

    if compound_score >= 0.1:
        return 'Positive', compound_score
    elif compound_score <= -0.1:
        return 'Negative', compound_score
    else:
        return 'Neutral', compound_score

def create_bar_plot(comments_df):
    sentiment_counts = comments_df['Sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/images/sentiment_distribution.png')

def create_time_series_plot(comments_df):
    comments_df['Date'] = pd.to_datetime(comments_df['Date'], errors='coerce')
    comments_df = comments_df.dropna(subset=['Date'])
    comments_df.set_index('Date', inplace=True)
    comments_over_time = comments_df.resample('D').size()
    plt.figure(figsize=(10, 6))
    comments_over_time.plot()
    plt.title('Number of Comments Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Comments')
    plt.tight_layout()
    plt.savefig('static/images/comments_over_time.png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display', methods=['GET', 'POST'])
def display():
    if request.method == 'POST':
        url = request.form['url']
        if is_valid_url(url):
            comments_df, article_title = fetch_comments(url)
            comments_df['Sentiment'], comments_df['Sentiment Score'] = zip(
                *comments_df['Comment'].apply(analyze_sentiment))
            comments_df.to_csv('comments.csv', index=False)
            create_bar_plot(comments_df)
            create_time_series_plot(comments_df)

            tables = [comments_df.to_html(classes='data', header="true", index=False)]
            return render_template('display.html', tables=tables, article_title=article_title)
        else:
            error = 'Invalid URL. Please enter a valid Hespress URL.'
            return render_template('index.html', error=error)
    return redirect(url_for('index'))

@app.route('/statistics')
def statistics():
    comments_df = pd.read_csv('comments.csv')
    total_comments = len(comments_df)
    positive_comments = len(comments_df[comments_df['Sentiment'] == 'Positive'])
    negative_comments = len(comments_df[comments_df['Sentiment'] == 'Negative'])
    neutral_comments = len(comments_df[comments_df['Sentiment'] == 'Neutral'])
    avg_sentiment_score = comments_df['Sentiment Score'].mean()

    return render_template('statistics.html',
                           total_comments=total_comments,
                           positive_comments=positive_comments,
                           negative_comments=negative_comments,
                           neutral_comments=neutral_comments,
                           avg_sentiment_score=avg_sentiment_score)

if __name__ == '__main__':
    app.run(debug=True)
