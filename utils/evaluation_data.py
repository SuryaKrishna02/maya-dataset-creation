import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from textstat import flesch_kincaid_grade, flesch_reading_ease

def calculate_complexity(text):
    try:
        fk_grade = flesch_kincaid_grade(text)
        f_ease = flesch_reading_ease(text)
        return fk_grade, f_ease
    except Exception:
        return 0, 0

def process_dataset(df, text_column):
    df['fk_grade'], df['f_ease'] = zip(*df[text_column].apply(calculate_complexity))
    df['length'] = df[text_column].apply(len)
    
    # Normalize complexity scores
    scaler = MinMaxScaler()
    df[['fk_grade_norm', 'f_ease_norm']] = scaler.fit_transform(df[['fk_grade', 'f_ease']])
    
    # Calculate overall complexity
    df['complexity'] = (df['fk_grade_norm'] + (1 - df['f_ease_norm'])) / 2
    
    return df

def create_bins(df, column, n_bins, use_qcut=True):
    if use_qcut:
        return pd.qcut(df[column], q=n_bins, labels=False, duplicates='drop')
    else:
        return pd.cut(df[column], bins=n_bins, labels=False)

def sample_texts(df, text_column, n_samples=20, n_bins=5, use_qcut=True):
    # Create complexity and length bins
    df['complexity_bin'] = create_bins(df, 'complexity', n_bins, use_qcut)
    df['length_bin'] = create_bins(df, 'length', n_bins, use_qcut)
    
    # Combine bins for stratification
    df['strata'] = df['complexity_bin'].astype(str) + '_' + df['length_bin'].astype(str)
    
    # Check for empty strata
    if len(df['strata'].unique()) < n_bins**2:
        print("Some strata are empty. Using complexity and length separately for stratification.")
        df['strata'] = df['complexity_bin'].astype(str) + df['length_bin'].astype(str)
    
    # Check for strata with very few samples
    strata_counts = df['strata'].value_counts()
    if (strata_counts < n_samples / (n_bins**2)).any():
        print("Warning: Some strata have very few samples. Consider reducing n_bins.")
    
    # Perform stratified sampling
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=42)
    for _, sample_idx in sss.split(df, df['strata']):
        sampled_data = df.iloc[sample_idx]
    
    # Sort the sampled data by complexity for a more organized output
    sampled_data_sorted = sampled_data.sort_values('complexity')
    
    # Select only the necessary columns
    result = sampled_data_sorted[[text_column, 'complexity', 'length', 'fk_grade', 'f_ease']]
    
    return result