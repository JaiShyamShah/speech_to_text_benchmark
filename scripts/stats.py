import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load CSV file into DataFrame
csv_file_path = r'C:\Users\jairo\OneDrive\Desktop\ionio\speech_to_text_benchmark\advanced_benchmark_results.csv'  # Replace with your actual CSV file path
result_df = pd.read_csv(csv_file_path)

# Display basic info about the DataFrame
print("DataFrame shape:", result_df.shape)
print("\nDataFrame columns:", result_df.columns.tolist())
print("\nFirst few rows:")
print(result_df.head())

# Check if required columns exist
required_columns = ['model', 'category', 'wer', 'insertions', 'deletions', 'substitutions']
missing_columns = [col for col in required_columns if col not in result_df.columns]

if missing_columns:
    print(f"\nWarning: Missing columns: {missing_columns}")
    print("Available columns:", result_df.columns.tolist())
else:
    # Compute average WER and error types per model and category
    avg_metrics = result_df.groupby(['model', 'category'])[['wer', 'insertions', 'deletions', 'substitutions']].mean().unstack()
    
    logger.info(f"Average metrics per model and category:\n{avg_metrics}")
    
    # Optional: Display the results in a more readable format
    print("\n" + "="*50)
    print("AVERAGE METRICS PER MODEL AND CATEGORY")
    print("="*50)
    print(avg_metrics)
    
    # Optional: Save results to CSV
    # avg_metrics.to_csv('average_metrics_results.csv')
    # print("\nResults saved to 'average_metrics_results.csv'")
    
    # Optional: Get summary statistics
    print("\n" + "="*30)
    print("SUMMARY STATISTICS")
    print("="*30)
    for metric in ['wer', 'insertions', 'deletions', 'substitutions']:
        if metric in result_df.columns:
            print(f"\n{metric.upper()}:")
            print(f"  Overall mean: {result_df[metric].mean():.4f}")
            print(f"  Overall std:  {result_df[metric].std():.4f}")
            print(f"  Min: {result_df[metric].min():.4f}")
            print(f"  Max: {result_df[metric].max():.4f}")