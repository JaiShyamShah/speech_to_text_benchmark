import pandas as pd
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_parakeet_results(csv_file_path=None):
    """Analyze Parakeet benchmark results and display comprehensive statistics."""
    
    # Use default path if none provided
    if csv_file_path is None:
        csv_file_path = r'C:\Users\jairo\OneDrive\Desktop\ionio\speech_to_text_benchmark\advanced_benchmark_results.csv'
    
    try:
        # Load CSV file into DataFrame
        result_df = pd.read_csv(csv_file_path)
        logger.info(f"Successfully loaded data from {csv_file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {csv_file_path}")
        return
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return
    
    # Display basic info about the DataFrame
    print("="*60)
    print("PARAKEET BENCHMARK RESULTS ANALYSIS")
    print("="*60)
    print(f"DataFrame shape: {result_df.shape}")
    print(f"Total samples analyzed: {len(result_df)}")
    print(f"DataFrame columns: {result_df.columns.tolist()}")
    
    # Check if required columns exist
    required_columns = ['model', 'category', 'wer', 'insertions', 'deletions', 'substitutions']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    
    if missing_columns:
        print(f"\nWarning: Missing columns: {missing_columns}")
        print("Available columns:", result_df.columns.tolist())
        return
    
    # Display first few rows
    print("\nFirst few rows:")
    print(result_df.head())
    
    # Check unique values
    print(f"\nUnique models: {result_df['model'].unique()}")
    print(f"Unique categories: {result_df['category'].unique()}")
    
    # Basic data validation
    print(f"\nData validation:")
    print(f"- Rows with missing WER: {result_df['wer'].isna().sum()}")
    print(f"- Rows with infinite WER: {np.isinf(result_df['wer']).sum()}")
    print(f"- Rows with empty predictions: {(result_df['pred_transcription'] == '').sum() if 'pred_transcription' in result_df.columns else 'N/A'}")
    
    # Clean data for analysis (remove infinite values)
    clean_df = result_df[~np.isinf(result_df['wer'])].copy()
    if len(clean_df) < len(result_df):
        print(f"- Removed {len(result_df) - len(clean_df)} rows with infinite WER for analysis")
    
    # Compute average WER and error types per model and category
    print("\n" + "="*70)
    print("AVERAGE METRICS PER MODEL AND CATEGORY")
    print("="*70)
    
    # Create pivot table for better formatting
    metrics_pivot = clean_df.groupby(['model', 'category'])[['wer', 'insertions', 'deletions', 'substitutions']].mean().unstack()
    
    # Format the output to match your desired style
    print(f"{'':15} {'wer':>15} {'insertions':>15} {'deletions':>15} {'substitutions':>15}")
    print(f"{'category':<15} {'clean':>7} {'noisy':>7} {'clean':>7} {'noisy':>7} {'clean':>8} {'noisy':>7} {'clean':>9} {'noisy':>7}")
    print(f"{'model':<15}")
    
    for model in clean_df['model'].unique():
        model_data = clean_df[clean_df['model'] == model]
        clean_data = model_data[model_data['category'] == 'clean']
        noisy_data = model_data[model_data['category'] == 'noisy']
        
        # Calculate averages
        clean_wer = clean_data['wer'].mean() if len(clean_data) > 0 else 0
        noisy_wer = noisy_data['wer'].mean() if len(noisy_data) > 0 else 0
        clean_ins = clean_data['insertions'].mean() if len(clean_data) > 0 else 0
        noisy_ins = noisy_data['insertions'].mean() if len(noisy_data) > 0 else 0
        clean_del = clean_data['deletions'].mean() if len(clean_data) > 0 else 0
        noisy_del = noisy_data['deletions'].mean() if len(noisy_data) > 0 else 0
        clean_sub = clean_data['substitutions'].mean() if len(clean_data) > 0 else 0
        noisy_sub = noisy_data['substitutions'].mean() if len(noisy_data) > 0 else 0
        
        print(f"{model:<15} {clean_wer:>7.6f} {noisy_wer:>7.6f} {clean_ins:>7.6f} {noisy_ins:>7.6f} {clean_del:>8.6f} {noisy_del:>7.6f} {clean_sub:>9.6f} {noisy_sub:>7.6f}")
    
    # Detailed statistics table
    print("\n" + "="*50)
    print("DETAILED STATISTICS TABLE")
    print("="*50)
    detailed_stats = clean_df.groupby(['model', 'category'])[['wer', 'insertions', 'deletions', 'substitutions']].agg(['mean', 'std', 'min', 'max']).round(6)
    print(detailed_stats)
    
    # Category comparison
    print("\n" + "="*40)
    print("PERFORMANCE BY CATEGORY")
    print("="*40)
    category_stats = clean_df.groupby('category')[['wer', 'insertions', 'deletions', 'substitutions']].agg(['mean', 'std', 'count']).round(6)
    print(category_stats)
    
    # Overall summary statistics
    print("\n" + "="*40)
    print("OVERALL SUMMARY STATISTICS")
    print("="*40)
    for metric in ['wer', 'insertions', 'deletions', 'substitutions']:
        if metric in clean_df.columns:
            print(f"\n{metric.upper()}:")
            print(f"  Overall mean: {clean_df[metric].mean():.6f}")
            print(f"  Overall std:  {clean_df[metric].std():.6f}")
            print(f"  Min: {clean_df[metric].min():.6f}")
            print(f"  Max: {clean_df[metric].max():.6f}")
            print(f"  Median: {clean_df[metric].median():.6f}")
    
    # Performance degradation analysis
    print("\n" + "="*50)
    print("NOISE IMPACT ANALYSIS")
    print("="*50)
    
    for model in clean_df['model'].unique():
        model_data = clean_df[clean_df['model'] == model]
        clean_wer = model_data[model_data['category'] == 'clean']['wer'].mean()
        noisy_wer = model_data[model_data['category'] == 'noisy']['wer'].mean()
        
        if not (np.isnan(clean_wer) or np.isnan(noisy_wer)):
            degradation = ((noisy_wer - clean_wer) / clean_wer) * 100 if clean_wer > 0 else 0
            print(f"{model}: Clean WER: {clean_wer:.6f}, Noisy WER: {noisy_wer:.6f}, Degradation: {degradation:.2f}%")
    
    # Save detailed results
    output_file = 'parakeet_detailed_stats.csv'
    try:
        detailed_stats.to_csv(output_file)
        print(f"\nDetailed statistics saved to: {output_file}")
    except Exception as e:
        logger.warning(f"Could not save detailed stats: {e}")
    
    # Sample predictions analysis
    if 'pred_transcription' in result_df.columns and 'true_transcription' in result_df.columns:
        print("\n" + "="*40)
        print("SAMPLE PREDICTIONS")
        print("="*40)
        
        # Show some examples
        for category in ['clean', 'noisy']:
            cat_data = result_df[result_df['category'] == category].head(3)
            print(f"\n{category.upper()} AUDIO EXAMPLES:")
            for idx, row in cat_data.iterrows():
                print(f"\nSample {idx}:")
                print(f"True: {row['true_transcription']}")
                print(f"Pred: {row['pred_transcription']}")
                print(f"WER:  {row['wer']:.4f}")

def main():
    """Main function to run the analysis."""
    # You can specify a custom path here
    csv_file_path = 'parakeet_benchmark_results.csv'  # Default path
    
    # Uncomment and modify the line below to use a custom path
    # csv_file_path = r'C:\Users\jairo\OneDrive\Desktop\ionio\speech_to_text_benchmark\parakeet_benchmark_results.csv'
    
    analyze_parakeet_results(csv_file_path)

if __name__ == "__main__":
    main()