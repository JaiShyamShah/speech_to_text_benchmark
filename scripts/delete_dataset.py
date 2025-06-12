from huggingface_hub import HfApi

# Initialize the Hugging Face API client
hf_api = HfApi()

# Define your dataset repository ID
repo_id = "jaishah2808/speech-to-text"  # Replace with your actual username and dataset name

try:
    # Delete the dataset repository
    hf_api.delete_repo(repo_id=repo_id, repo_type="dataset")
    print(f"Successfully deleted dataset {repo_id}")
except Exception as e:
    print(f"Error deleting dataset: {str(e)}")