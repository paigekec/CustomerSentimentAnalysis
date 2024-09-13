# Importing necessary libraries for the project

# For data manipulation and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# For text processing and sentiment analysis
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Loading Spacy English model
nlp = spacy.load("en_core_web_sm")

# Sample function to clean and preprocess the text
def preprocess_text(text):
    doc = nlp(text.lower())  # Converting text to lowercase and tokenizing
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]  # Lemmatization and stopword removal
    return " ".join(tokens)

# Sample dataset of customer reviews
data = {
    'review': [
        "The product was great, I loved it!",
        "Terrible experience, would not recommend.",
        "The customer service was amazing, but the product quality was mediocre.",
        "Fast shipping, but the item was damaged on arrival.",
        "Absolutely fantastic! Exceeded my expectations.",
        "Not what I expected. The description was misleading.",
        "Decent product for the price, but there are better options out there."
    ],
    'date': pd.date_range(start='2023-01-01', periods=7, freq='M')
}

# Converting to a pandas dataframe
df = pd.DataFrame(data)

# Preprocessing the reviews
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Applying sentiment analysis to the dataset
df['sentiment'] = df['review'].apply(analyze_sentiment)

# Function to visualize the sentiment distribution
def visualize_sentiment(df):
    plt.figure(figsize=(8,5))
    sns.countplot(x='sentiment', data=df, palette="coolwarm")
    plt.title("Sentiment Distribution of Reviews")
    plt.show()

# Function to generate a word cloud of the reviews
def generate_wordcloud(df):
    text = " ".join(review for review in df['cleaned_review'])
    wordcloud = WordCloud(background_color="white").generate(text)
    
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Implementing topic modeling using LDA (Latent Dirichlet Allocation)
def topic_modeling(df, num_topics=3):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df['cleaned_review'])

    LDA = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    LDA.fit(doc_term_matrix)

    # Displaying the top words in each topic
    for index, topic in enumerate(LDA.components_):
        print(f"TOP 10 WORDS FOR TOPIC #{index}")
        print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
        print("\n")

# Visualizing the sentiment distribution
visualize_sentiment(df)

# Generating a word cloud for the reviews
generate_wordcloud(df)

# Performing topic modeling
topic_modeling(df)
