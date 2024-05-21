import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def fetch_article_title(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').get_text()
        return title
    except Exception as e:
        return str(e)

@app.route('/display', methods=['POST'])
def display():
    url = request.form['url']
    article_title = fetch_article_title(url)
    try:
        comments_df = pd.read_csv('comments.csv')  # Replace with actual data fetching logic
        comments = comments_df.to_dict(orient='records')
        return render_template('display.html', article_title=article_title, comments=comments)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/download_csv')
def download_csv():
    return send_file('comments.csv', as_attachment=True)

@app.route('/statistics')
def statistics():
    comments_df = pd.read_csv('comments.csv')  # Replace with the actual DataFrame source

    total_comments = len(comments_df)
    positive_comments = len(comments_df[comments_df['Sentiment'] == 'Positive'])
    negative_comments = len(comments_df[comments_df['Sentiment'] == 'Negative'])
    neutral_comments = len(comments_df[comments_df['Sentiment'] == 'Neutral'])
    avg_sentiment_score = comments_df['Sentiment Score'].mean()

    create_visualizations(comments_df)

    return render_template('statistics.html',
                           total_comments=total_comments,
                           positive_comments=positive_comments,
                           negative_comments=negative_comments,
                           neutral_comments=neutral_comments,
                           avg_sentiment_score=avg_sentiment_score)

def create_visualizations(comments_df):
    sentiment_counts = comments_df['Sentiment'].value_counts()

    plt.figure(figsize=(4, 3))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('static/images/sentiment_distribution_pie.png')
    plt.close()

    likes_per_user = comments_df.groupby('User Name')['Likes'].sum().reset_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x='User Name', y='Likes', data=likes_per_user)
    plt.title('Likes per User')
    plt.xlabel('User Name')
    plt.ylabel('Number of Likes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/images/likes_per_user_bar.png')
    plt.close()

    comments_df['Date'] = pd.to_datetime(comments_df['Date'])
    comments_by_date = comments_df.groupby(comments_df['Date'].dt.date).size()
    plt.figure(figsize=(4, 3))
    plt.plot(comments_by_date.index, comments_by_date.values, marker='o', linestyle='-')
    plt.title('Number of Comments over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/images/comments_over_time.png')
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
