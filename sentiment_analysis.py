import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import requests
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
import nltk
from mtranslate import translate
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('stopwords')
nltk.download('vader_lexicon')

def is_valid_url(url):
    try:
        result = urlparse(url)
        if all([result.scheme, result.netloc]):
            if result.netloc == "www.hespress.com":
                return True
            else:
                return False
        else:
            return False
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

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
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
    df.to_csv('comments.csv', index=False)  # Save to CSV
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

def main():
    url = input("Enter a valid Hespress URL: ")
    if is_valid_url(url):
        comments_df, article_title = fetch_comments(url)
        comments_df['Sentiment'], comments_df['Sentiment Score'] = zip(
            *comments_df['Comment'].apply(analyze_sentiment))
        comments_df.to_csv('comments.csv', index=False)  # Save the DataFrame to a CSV file
        print(f"Comments for '{article_title}' have been fetched and analyzed.")

        # Statistics
        total_comments = len(comments_df)
        positive_comments = comments_df[comments_df['Sentiment'] == 'Positive']
        negative_comments = comments_df[comments_df['Sentiment'] == 'Negative']
        neutral_comments = comments_df[comments_df['Sentiment'] == 'Neutral']
        avg_sentiment_score = comments_df['Sentiment Score'].mean()

        print(f"Total Comments: {total_comments}")
        print(f"Positive Comments: {len(positive_comments)}")
        print(f"Negative Comments: {len(negative_comments)}")
        print(f"Neutral Comments: {len(neutral_comments)}")
        print(f"Average Sentiment Score: {avg_sentiment_score}")

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Sentiment', data=comments_df)
        plt.title('Sentiment Distribution')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(comments_df['Sentiment Score'], kde=True)
        plt.title('Sentiment Score Distribution')
        plt.show()

    else:
        print("Invalid URL. Please enter a valid Hespress URL.")

if __name__ == "__main__":
    main()
