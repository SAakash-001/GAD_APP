import kaggle

# Download and unzip Common Voice dataset from Kaggle
kaggle.api.dataset_download_files('mozillaorg/common-voice', path='data', unzip=True)
