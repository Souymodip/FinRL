import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm  # Import tqdm for progress bar
import os

def get_model():
    # Create the sentiment-analysis pipeline with FinBERT
    model_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    # If the macOS MPS backend is available, move the model to the MPS device for acceleration.
    if torch.backends.mps.is_available():
        model_pipeline.model.to("mps")
    print(f'Loaded Model :{type(model_pipeline)}. Device: {model_pipeline.device}')
    return model_pipeline


def get_sentiment_scores(model, articles, batch_size):
    """
    Processes articles in batches for sentiment analysis with a progress bar.

    Parameters:
    - model: the sentiment analysis pipeline
    - articles: list of text articles (headlines)
    - batch_size: number of articles to process in one batch

    Returns:
    - A list of sentiment analysis results where each result is a dictionary with keys 'label' and 'score'.
    """
    results = []
    # Iterate using tqdm for a progress bar over the batches
    for i in tqdm(range(0, len(articles), batch_size), desc="Processing batches"):
        batch = articles[i:i + batch_size]
        batch_results = model(batch)
        results.extend(batch_results)
    return results


def load_data_frame(csv_path):
    try:
        df = pd.read_csv(
            csv_path,
            on_bad_lines='skip',  # Skip problematic lines
            engine='python',
            dtype=str,
        )
        # Convert any date-like columns if they exist
        for date in ['Date', 'date']:
            if date in df.columns:
                df[date] = pd.to_datetime(df[date], errors='coerce')
        # Drop the 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

        # Ensure all object columns are treated as strings
        object_columns = df.select_dtypes(include=['object']).columns
        df[object_columns] = df[object_columns].astype('string')

        print("Successfully loaded DataFrame!")
        df.info()
        print("Columns:", df.columns.to_list())
        print("Shape:", df.shape)

        return df

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
        print("Check the CSV file for quoting/escaping issues.")
    except ValueError as e:
        print(f"ValueError: {e}")
        print("There might be issues with data types or parsing.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def add_sentiment_analysis(model, df, col_name, output_csv):    
    if df is None or col_name not in df.columns:
        print(f"DataFrame is empty or missing a [{col_name}] column.")
        return

    # Extract all headlines from the 'title' column into a list.
    headlines = df[col_name].tolist()
    
    # Define your desired batch size (you can adjust this parameter)
    batch_size = 16  # For example, process 16 headlines at a time

    # Get sentiment scores in batches with a progress bar
    sentiment_results = get_sentiment_scores(model, headlines, batch_size=batch_size)
    
    # Add new columns 'sentiment_label' and 'sentiment_score' to the DataFrame
    df['sentiment_label'] = [res['label'] for res in sentiment_results]
    df['sentiment_score'] = [res['score'] for res in sentiment_results]
    
    # Optional: Save the DataFrame with sentiment analysis results to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Sentiment analysis completed. Results saved to: {output_csv}")


def sentiment_main():
    csv_path = '/Users/souymodip/GIT/FinRL/news/raw_partner_headlines.csv'
    assert os.path.exists(csv_path), f'[{csv_path}] doesn\'t not exists'
    output_csv = csv_path.split('.')[0] +"_sentiment_with_FinBert.csv"

    col_name = 'headline'
    df = load_data_frame(csv_path=csv_path)
    assert col_name in df.columns, f'columns: {df.columns.to_list()}'
    # test_df = df[:100]

    model = get_model()
    add_sentiment_analysis(model=model, df=df, col_name=col_name, output_csv=output_csv)


def get_sectors():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    sp500_sectors = df.groupby('GICS Sector')['Symbol'].apply(list).to_dict()
    return sp500_sectors



def organize_sentiment():
    csv_path = '/Users/souymodip/GIT/FinRL/news/analyst_ratings_processed.csv'
    csv_path = csv_path.split('.')[0] +"_sentiment_with_FinBert.csv"
    assert os.path.exists(csv_path), f'[{csv_path}] doesn\'t not exists'
    df = load_data_frame(csv_path=csv_path)
    assert set(['date', 'stock', 'sentiment_label', 'sentiment_score']).issubset(df.columns.to_list()), f'Columns: {df.columns.to_list()}'
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')

    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df['date'] = df['date'].dt.tz_localize(None)
    df['date'] =df['date'].dt.normalize()
    
    grouped = df.groupby(['date', 'stock', 'sentiment_label'])['sentiment_score'].sum()

    df_result = grouped.unstack(fill_value=0)
    df_result = df_result[['neutral', 'positive', 'negative']]
    df_result.rename(columns={
        'neutral': 'neutral_score',
        'positive': 'positive_score',
        'negative': 'negative_score'
    }, inplace=True)
    df_result = df_result.reset_index()
    return df_result



def merge_sentiment(df_history, df_sentiment):
    df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'], utc=True, errors='coerce')
    # df_sentiment['date'] = pd.to_datetime(df_sentiment['date'], errors='coerce')
    
    df_history['date'] = df_history['date'].dt.tz_localize(None)
    df_sentiment['date'] = df_sentiment['date'].dt.tz_localize(None)

    print("\t\tdf_history date dtype:", df_history['date'].dtype)
    print("\t\tdf_sentiment date dtype:", df_sentiment['date'].dtype)


    df_history['tic'] = df_history['tic'].astype('string')
    df_sentiment['stock'] = df_sentiment['stock'].astype('string')

    assert df_history['date'].dtype == df_sentiment['date'].dtype, f'df_history: {df_history['date'].dtype} ~ df_sentiment: {df_sentiment['date'].dtype}'
    assert df_history['tic'].dtype == df_sentiment['stock'].dtype, f'df_history: {df_history['tic'].dtype} ~ df_sentiment: {df_sentiment['stock'].dtype}'

    # Use 'left' join to retain all records from df_history even if there is no sentiment info

    print(df_sentiment[df_sentiment['stock'] == 'ADBE']['neutral_score'].head())
    print(df_history[df_history['tic'] == 'ADBE']['close'].head())

    merged_df = pd.merge(df_history, df_sentiment, left_on=['date', 'tic'], right_on=['date', 'stock'],  how='left')


    # Optionally, drop the redundant 'stock' column (since 'tic' already contains the ticker info)
    merged_df.drop('stock', axis=1, inplace=True)
    sentiment_columns = ['neutral_score', 'positive_score', 'negative_score']
    
    print(merged_df.columns)
    print(merged_df.dtypes)
    # import pdb; pdb.set_trace()
    merged_df[sentiment_columns] = merged_df[sentiment_columns].fillna(0)

    # merged_df.set_index('date', inplace=True)
    # merged_df.sort_index(inplace=True)
    # merged_df[sentiment_columns] = merged_df[sentiment_columns].interpolate(method='time')
    # merged_df.reset_index(inplace=True)

    # import pdb; pdb.set_trace() 
    return merged_df


if __name__ == "__main__":
    # Set the path to your CSV file here
    # organize_main()
    sect_dict = get_sectors()
    print(f'Sectors: {sect_dict.keys()}')
    sect_it = sect_dict['Information Technology']
    df_sentiment = organize_sentiment()
    import pdb; pdb.set_trace()
    oldest_row = df_sentiment.loc[df_sentiment['date'].idxmin()]
    
