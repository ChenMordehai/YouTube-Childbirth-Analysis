#!/usr/bin/env python
"""
Professional YouTube Data Analysis Script
===========================================
TODO: add description here
"""

import os
import re
from collections import Counter
from datetime import datetime, timedelta
from io import BytesIO
from unlimited_classifier.scorer import Scorer
from unlimited_classifier import TextClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
import spacy
from deepface import DeepFace
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from transformers import (T5ForConditionalGeneration, T5Tokenizer,
                          pipeline, AutoTokenizer, AutoModelForCausalLM)
from langdetect import detect, DetectorFactory

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DetectorFactory.seed = 0
from tqdm import tqdm

tqdm.pandas()

import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

# -----------------------------------------------------------------------------
# Define devices for torch and pipelines
# -----------------------------------------------------------------------------
# For torch operations: use torch.device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
# For transformers pipelines: pass integer device id (>=0 for GPU, -1 for CPU)
pipeline_device = 0 if torch.cuda.is_available() else -1
if spacy.prefer_gpu():
    print("\nUsing spacy with GPU\n")
nlp_sent = spacy.blank("en")
nlp_sent.add_pipe("sentencizer")
# -----------------------------------------------------------------------------
# 1. Utility Functions
# -----------------------------------------------------------------------------
# Create an 'images' directory if it does not exist
images_dir = "/sise/home/mordeche/bigdata_youtube/analysis/images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


def save_and_show_plot(filename):
    """
    Save the current plot to the images/ directory and display it.
    """
    plt.tight_layout()
    save_path = os.path.join(images_dir, filename)
    plt.savefig(save_path)
    print(f"Plot saved: {save_path}")
    plt.show()


def parse_relative_time(relative_time):
    """
    Convert relative time strings like '1 year ago' into a datetime object.
    """
    now = datetime.now()
    match = re.match(r'(\d+)\s+(\w+)\s+ago', relative_time)
    if not match:
        return None
    value, unit = int(match.group(1)), match.group(2)
    if 'day' in unit:
        delta = timedelta(days=value)
    elif 'month' in unit:
        delta = timedelta(days=30 * value)  # Approximate month as 30 days
    elif 'year' in unit:
        delta = timedelta(days=365 * value)  # Approximate year as 365 days
    else:
        return None
    return now - delta


def parse_duration(duration_str):
    """
    Convert a duration string in HH:MM:SS, MM:SS, or H:MM format to minutes.
    """
    parts = duration_str.split(':')
    total_seconds = 0
    try:
        parts = [int(x) for x in parts if x.strip()]
        parts.reverse()  # seconds, minutes, hours
        if len(parts) > 0:
            total_seconds += parts[0]
        if len(parts) > 1:
            total_seconds += parts[1] * 60
        if len(parts) > 2:
            total_seconds += parts[2] * 3600
    except ValueError:
        print(f"Error parsing duration: {duration_str}")
        return None
    return total_seconds / 60


def numeric_views(view_str):
    """
    Remove non-numeric characters (except commas) and convert to integer.
    """
    try:
        cleaned_str = ''.join(filter(lambda x: x.isdigit() or x == ',', view_str))
        cleaned_str = cleaned_str.replace(',', '')
        return int(cleaned_str)
    except ValueError:
        return None


def get_sentiment(text):
    """
    Return the sentiment polarity of the provided text.
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


def display_topics_graph(model, feature_names, no_top_words):
    """
    Display the top words for each LDA topic as a horizontal bar chart.
    """
    num_topics = model.components_.shape[0]
    fig, axes = plt.subplots(num_topics, 1, figsize=(10, 4 * num_topics))
    if num_topics == 1:
        axes = [axes]
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = topic[top_indices]
        ax = axes[topic_idx]
        sns.barplot(x=top_weights, y=top_words, palette="viridis", ax=ax)
        ax.set_title(f"Topic {topic_idx}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Word")
    plt.tight_layout()
    save_and_show_plot('transcribe_topic_modeling.png')


def fetch_image(url):
    """
    Download an image from the given URL and return a PIL image.
    """
    try:
        response = requests.get(url, timeout=20)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"Error fetching image {url}: {e}")
        return None


classifier_medical_advice_correct_false_ambiguous = TextClassifier(
    labels=[
        'correct clinical advice',
        'ambiguous clinical advice',
        'false clinical advice'
    ],
    model="t5-base",
    tokenizer="t5-base",
    scorer=Scorer(),
    # device=torch_device,
    # device_map=torch_device
)


def score_recommendation(text):
    """
    Score a recommendation sentence
    """
    try:
        output = classifier_medical_advice_correct_false_ambiguous.invoke(text)
        return output[0][0]
    except Exception as e:
        return None


is_clinical_advice_classifier = TextClassifier(
    labels=[
        'clinical advice',
        'non clinical advice'
    ],
    model="t5-base",
    tokenizer="t5-base",
    scorer=Scorer(),
    # device=torch_device,
    # device_map=torch_device
)

def fast_sentence_split(text):
    """
    Splits input text into sentences using spaCy's sentencizer.

    Args:
        text (str): The text to split.

    Returns:
        List[str]: A list of sentence strings.
    """
    doc = nlp_sent(text)
    return [sent.text.strip() for sent in doc.sents]

def extract_recommendations_with_classifier(transcript, threshold=0.8):
    if not transcript:
        return []
    # sentences = sent_tokenize(transcript)
    sentences = fast_sentence_split(transcript)
    recommendations = []
    for sent in sentences[:20]:
        sent_len = len(sent.split())
        if sent_len >= 15 and sent_len < 100000:
            try:
                output = is_clinical_advice_classifier.invoke(sent)
                class_, score = output[0]
                if class_ == 'clinical advice' and score >= threshold:
                    recommendations.append(sent)
            except Exception as e:
                return None
    return recommendations

def score_video_recommendations(transcript):
    """
    For a given video transcript, extract recommendation sentences and score each using FlanT5.
    Returns a dictionary with counts.
    """
    recommendations = extract_recommendations_with_classifier(transcript)
    counts = {"correct clinical advice": 0, "ambiguous clinical advice": 0,
              "false clinical advice": 0, "total": len(recommendations)}
    for rec in recommendations:
        label = score_recommendation(rec)
        if label in counts:
            counts[label] += 1
    return counts


classifier_medication_non_medication = TextClassifier(
    labels=[
        'medication',
        'no medication'
    ],
    model="t5-base",
    tokenizer="t5-base",
    scorer=Scorer(),
    # device=torch_device,
    # device_map=torch_device
)


def extract_medications_med7(text, nlp_med7, chunk_size=10000):
    meds = []
    # Process the text in chunks of chunk_size characters
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        doc = nlp_med7(chunk)
        for ent in doc.ents:
            if ent.label_ == 'DRUG' and len(ent.text) > 5 and len(ent.text) < 20:
                meds.append(ent.text)
    return meds


# -----------------------------------------------------------------------------
# 2. Main Analysis Workflow
# -----------------------------------------------------------------------------

def main():
    # Load dataset
    print("loading data")
    df = pd.read_csv('/sise/home/mordeche/bigdata_youtube/data/concat_transcribe_dfs.csv')
    # df = df.sample(n=10)
    df = df.fillna('')
    df['duration_minutes'] = df['duration'].apply(parse_duration)
    df = df[df['duration_minutes'] >= 0.25]

    def is_english(text):
        try:
            return detect(text) == 'en'
        except Exception:
            return False

    df = df[df['transcription'].apply(is_english)]
    df['published_datetime'] = df['published_time'].apply(parse_relative_time)
    df['views'] = df['views'].apply(numeric_views)

    # 2.1 Distribution of Videos per Query
    # Step 1: Identify the top N hashtags
    top_n = 50
    top_hashtags = df['hashtag'].value_counts().nlargest(top_n).index
    # Step 2: Create a new column with 'Other' for less frequent hashtags
    df['hashtag_modified'] = df['hashtag'].where(df['hashtag'].isin(top_hashtags), 'Other')
    # Step 3: Plot the data
    plt.figure(figsize=(12, 12))
    order = df['hashtag_modified'].value_counts().index
    order = order[order != 'Other']
    sns.countplot(y='hashtag_modified', data=df[df['hashtag_modified'] != 'Other'], order=order, palette="coolwarm")
    plt.title('Number of Videos per Query')
    plt.xlabel('Number of Videos')
    plt.ylabel('Query')
    plt.tight_layout()
    save_and_show_plot('videos_per_query.png')


    # 2.2 Video Length Distribution
    bins = [0, 10, 30, 60, 120, float('inf')]
    labels = ['<10 min', '10-30 min', '30-60 min', '60-120 min', '>120 min']
    df['duration_category'] = pd.cut(df['duration_minutes'], bins=bins, labels=labels, right=False)
    duration_counts = df['duration_category'].value_counts().reindex(labels)
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x=duration_counts.index, y=duration_counts.values, palette="coolwarm")
    plt.title('Distribution of Video Length (Minutes)')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Number of Videos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_and_show_plot('videos_duration.png')

    # 2.3 Views per Query Distribution
    # top_queries = df['hashtag'].value_counts().head(6).index
    # df_top = df[df['hashtag'].isin(top_queries)]
    # g = sns.FacetGrid(df_top, col="hashtag", col_wrap=3, height=4, sharex=False, sharey=False)
    # g.map(sns.histplot, "views", bins=10, color='mediumseagreen')
    # g.fig.suptitle('Views Distribution per Query (10 bins)', y=1.02)
    # save_and_show_plot('views_per_query.png')
    hashtag_stats = df.groupby('hashtag')['views'].agg(['sum', 'count']).reset_index()
    hashtag_stats.columns = ['hashtag', 'total_views', 'hashtag_count']
    top_n = 10
    top_hashtags = hashtag_stats.nlargest(top_n, 'total_views')
    other_stats = hashtag_stats[~hashtag_stats['hashtag'].isin(top_hashtags['hashtag'])]
    other_summary = pd.DataFrame({
        'hashtag': ['Other'],
        'total_views': [other_stats['total_views'].sum()],
        'hashtag_count': [other_stats['hashtag_count'].sum()]
    })
    plot_data = pd.concat([top_hashtags, other_summary])
    fig, ax1 = plt.subplots(figsize=(12, 8))
    hashtags = plot_data['hashtag']
    bar_width = 0.4
    x = np.arange(len(hashtags))

    bars1 = ax1.bar(x - bar_width / 2, plot_data['total_views'], bar_width, label='Total Views', color='b')
    ax1.set_xlabel('Hashtags')
    ax1.set_ylabel('Total Views', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticks(x)
    ax1.set_xticklabels(hashtags, rotation=45, ha='right')
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + bar_width / 2, plot_data['hashtag_count'], bar_width, label='Hashtag Count', color='g')
    ax2.set_ylabel('Hashtag Count', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title('Top 10 Hashtags by Total Views with Hashtag Count')
    fig.tight_layout()
    save_and_show_plot('views_per_query.png')


    # 2.4 Publication Years Distribution
    df['year'] = df['published_datetime'].dt.year.fillna(-1).astype(int)
    df['year'] = df['year'].replace(-1, pd.NA)
    plt.figure(figsize=(12, 6))
    order_year = df['year'].value_counts().sort_index().index
    sns.countplot(x='year', data=df, order=order_year, palette="muted")
    plt.title('Distribution of Video Publication Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Videos')
    save_and_show_plot('years_dist.png')

    # 2.5 Topic Modeling on Transcriptions
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(df['transcription'].dropna())
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)
    display_topics_graph(lda, vectorizer.get_feature_names_out(), no_top_words=10)

    # 2.6 Sentiment Analysis on Transcriptions
    print("\nGetting sentiment\n")
    df['sentiment'] = df['transcription'].progress_apply(get_sentiment)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], bins=30, kde=True, color='orchid')
    plt.title('Sentiment Distribution of Transcriptions')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    save_and_show_plot('sentiment_transcription_distribution.png')

    positive = (df['sentiment'] > 0).sum()
    negative = (df['sentiment'] < 0).sum()
    neutral = (df['sentiment'] == 0).sum()
    sentiment_counts = {'Positive': positive, 'Negative': negative, 'Neutral': neutral}
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()), palette="pastel")
    plt.title('Total Sentiment Counts in Transcriptions')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    save_and_show_plot('total_transcription_sentiment_counts.png')

    # 2.7 Unique Channels and Videos per Channel
    unique_channels = df['channel_id'].nunique()
    print(f"Unique Channels: {unique_channels}")
    channel_counts = df['channel_name'].value_counts()
    top_n = 10
    top_channels = channel_counts.head(top_n)
    other_count = channel_counts[top_n:].sum()
    other_series = pd.Series({'Other': other_count})
    channel_counts_filtered = pd.concat([top_channels, other_series])
    plt.figure(figsize=(15, 7))
    channel_counts_filtered.plot(kind='bar', color='skyblue')
    plt.title('Number of Videos per Channel (Top 10 + Other)')
    plt.xlabel('Channel')
    plt.ylabel('Number of Videos')
    plt.xticks(rotation=45)
    save_and_show_plot('channel_counts_filtered.png')

    # 2.8 Medical vs. Personal Classification via unlimited_classifier.TextClassifier
    def classify_transcription_medical_personal(text):
        try:
            torch.cuda.empty_cache()
            output = classifier_medical_or_personal_or_other.invoke(text)
            return output[0][0]
        except Exception as e:
            print(f"Error processing transcription: {e}")
            return None


    classifier_medical_or_personal_or_other = TextClassifier(
        labels=[
            'Personal Experience',
            'Medical Recommendation',
            'Other'
        ],
        model="t5-base",
        tokenizer="t5-base",
        scorer=Scorer(),
        # device=torch_device,
        # device_map=torch_device
    )
    print("Classifing transcription medical personal")
    df['classification_medical_personal'] = df['transcription'].progress_apply(classify_transcription_medical_personal)
    classification_medical_personal_counts = df['classification_medical_personal'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=classification_medical_personal_counts.index,
                y=classification_medical_personal_counts.values,
                palette="Set2")
    plt.title('Medical vs. Personal Classification Counts')
    plt.xlabel('Classification')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    save_and_show_plot('classification_medical_personal_counts.png')

    # 2.9 Pros and Cons Classification on Medical Recommendations
    def classify_medical_recommendation(text):
        try:
            output = classify_medical_recommendation_pros_n_cons.invoke(text)
            return output[0][0]
        except Exception as e:
            print(f"Error processing transcription: {e}")
            return None

    classify_medical_recommendation_pros_n_cons = TextClassifier(
        labels=[
            'balanced',
            'only pros',
            'only cons'
        ],
        model="t5-base",
        tokenizer="t5-base",
        scorer=Scorer(),
        device=torch_device,
        device_map=torch_device
    )
    medical_recommendations = df[df['classification_medical_personal'].str.lower() == 'medical recommendation'].copy()
    print("\npros_cons_classification\n")
    medical_recommendations['pros_cons_classification'] = medical_recommendations['transcription'].progress_apply(
        classify_medical_recommendation)
    pros_cons_counts = medical_recommendations['pros_cons_classification'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=pros_cons_counts.index, y=pros_cons_counts.values, palette="Set3")
    plt.title('Pros and Cons Classification Counts')
    plt.xlabel('Classification')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    save_and_show_plot("prons_and_cons_classification_count.png")

    # 2.10 Topic Modeling on Queries using Sentence Embeddings
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    queries_embeddings = sentence_model.encode(df['hashtag'].tolist())
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['query_topic'] = kmeans.fit_predict(queries_embeddings)
    topic_counts = df['query_topic'].value_counts()
    print("Query Topics Count:")
    print(topic_counts)
    topic_views = df.groupby('query_topic')['views'].agg(['sum', 'mean', 'count'])
    print("Views and Videos per Query Topic:")
    print(topic_views)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=topic_views.index, y=topic_views['sum'], palette="rocket")
    plt.title('Total Views per Query Topic')
    plt.xlabel('Topic')
    plt.ylabel('Total Views')
    save_and_show_plot('total_views_per_topic_embeddings.png')
    plt.figure(figsize=(10, 6))
    sns.barplot(x=topic_views.index, y=topic_views['count'], palette="mako")
    plt.title('Number of Videos per Query Topic')
    plt.xlabel('Topic')
    plt.ylabel('Number of Videos')
    save_and_show_plot('total_views_per_topic__embedding_count.png')
    df.to_csv('after_total_views_per_topic_embedding_count.csv', index=False)
    # 2.11 Thumbnail Image Analysis: Description & Sentiment
    # captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning",
    #                      batch_size=32, device=pipeline_device, use_fast=True)
    # # First, fetch all thumbnail images into a list
    # thumbnail_urls = df['thumbnail_url'].tolist()
    # thumbnails = [fetch_image(url) for url in thumbnail_urls]
    # # Keep track of valid image indices (where fetch_image succeeded)
    # valid_indices = [i for i, img in enumerate(thumbnails) if img is not None]
    # valid_thumbnails = [img for img in thumbnails if img is not None]
    #
    # # Process all valid images in a batch using the captioner pipeline
    # results = captioner(valid_thumbnails)
    # # Initialize a list for all results with default (None, None) for failures
    # thumbnail_results = [(None, None)] * len(thumbnails)
    # for idx, res in zip(valid_indices, results):
    #     caption = res[0]['generated_text']
    #     sentiment = TextBlob(caption).sentiment.polarity
    #     thumbnail_results[idx] = (caption, sentiment)
    # df_thumbnail = pd.DataFrame(thumbnail_results, columns=['thumbnail_description', 'thumbnail_sentiment'])
    def analyze_thumbnail_images(df, batch_size=32):
        """
        Analyze thumbnail images by generating descriptions and sentiments in batches.

        Parameters:
        - df: pandas DataFrame containing a 'thumbnail_url' column.
        - batch_size: Number of images to process in each batch.

        Returns:
        - Updated DataFrame with 'thumbnail_description' and 'thumbnail_sentiment' columns.
        """
        print("Loading image captioning model...")
        captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning",batch_size=32,
                             device=pipeline_device, use_fast=True)

        thumbnail_urls = df['thumbnail_url'].tolist()
        thumbnail_descriptions = [None] * len(thumbnail_urls)
        thumbnail_sentiments = [None] * len(thumbnail_urls)

        for start_idx in tqdm(range(0, len(thumbnail_urls), batch_size)):
            end_idx = start_idx + batch_size
            batch_urls = thumbnail_urls[start_idx:end_idx]
            batch_images = [fetch_image(url) for url in batch_urls]

            valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
            valid_images = [img for img in batch_images if img is not None]

            if not valid_images:
                continue

            results = captioner(valid_images)

            for idx, res in zip(valid_indices, results):
                caption = res[0]['generated_text']
                sentiment = TextBlob(caption).sentiment.polarity
                thumbnail_descriptions[start_idx + idx] = caption
                thumbnail_sentiments[start_idx + idx] = sentiment
        return thumbnail_descriptions, thumbnail_sentiments
    thumbnail_descriptions, thumbnail_sentiments = analyze_thumbnail_images(df)
    df['thumbnail_description'] = thumbnail_descriptions
    df['thumbnail_sentiment'] = thumbnail_sentiments

    plt.figure(figsize=(10, 6))
    sns.histplot(df['thumbnail_sentiment'].dropna(), bins=20, kde=True, color='teal')
    plt.title('Sentiment Distribution of Thumbnail Descriptions')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    save_and_show_plot('thumbnail_sentiment_distribution.png')
    df.to_csv('after_thumbnail_sentiment_distribution.csv', index=False)

    # 2.12 Scoring Medical Recommendations in Transcripts
    def score_video_recommendations_wrapper(transcript):
        return score_video_recommendations(transcript)

    recommendation_scores = df['transcription'].progress_apply(score_video_recommendations_wrapper)
    df['recommendation_scores'] = recommendation_scores
    df['num_recommendations'] = df['recommendation_scores'].apply(lambda x: x.get('total', 0))
    df['num_correct'] = df['recommendation_scores'].apply(lambda x: x.get('correct clinical advice', 0))
    df['num_ambiguous'] = df['recommendation_scores'].apply(lambda x: x.get('ambiguous clinical advice', 0))
    df['num_false'] = df['recommendation_scores'].apply(lambda x: x.get('false clinical advice', 0))
    total_recs = df['num_recommendations'].sum()
    total_correct = df['num_correct'].sum()
    total_ambiguous = df['num_ambiguous'].sum()
    total_false = df['num_false'].sum()
    print("Overall Clinical Recommendation Scores:")
    print(f"Total Clinical Recommendations: {total_recs}")
    print(f"Correct: {total_correct}")
    print(f"Ambiguous: {total_ambiguous}")
    print(f"False: {total_false}")

    df.to_csv('after_score_video_recommendations.csv', index=False)

    # 2.13 Medication Extraction from Transcripts using Med7
    if spacy.prefer_gpu():
        print("\nUsing spacy with GPU\n")
    med7 = spacy.load("/sise/home/mordeche/bigdata_youtube/analysis/med7_spacy/content/med7_spacy")

    df['extracted_medications'] = df['transcription'].progress_apply(
        lambda text: extract_medications_med7(text, med7)
    )
    all_medications = [med.lower() for meds in df['extracted_medications'] for med in meds]
    medications_counts = Counter(all_medications)
    print("Medications Frequency:")
    print(medications_counts)
    if medications_counts:
        meds_df = pd.DataFrame(medications_counts.items(), columns=["Medication", "Count"]).sort_values(by="Count",
                                                                                                        ascending=False)
        top_10 = meds_df.head(10)
        # other_count = meds_df.iloc[10:]['Count'].sum()
        # other_df = pd.DataFrame([{"Medication": "other", "Count": other_count}])
        # final_df = pd.concat([top_10, other_df], ignore_index=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Count", y="Medication", data=top_10, palette="Blues_d")
        plt.title("Popular Medications Mentioned in Videos")
        plt.xlabel("Frequency")
        plt.ylabel("Medication")
        save_and_show_plot("popular_medications.png")

    df.to_csv('after_analysis.csv', index=False)


if __name__ == '__main__':
    main()
