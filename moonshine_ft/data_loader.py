import os
import logging
from datasets import load_from_disk, Audio

class DataLoader:
    def __init__(self, dataset_path, text_column="sentence", cache_dir="/kaggle/working/dataset_cache"):
        """
        Initializes the loader for local Hugging Face datasets.
        
        :param dataset_path: Path to the root directory (contains 'train'/'test' folders)
        :param text_column: The name of the column containing the transcriptions
        :param cache_dir: Writable directory for processing files
        """
        self.dataset_path = dataset_path
        self.text_column = text_column
        self.cache_dir = cache_dir
        self.logger = self.setup_logging()
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def load_dataset(self):
        """
        Loads the local arrow dataset and prepares it for Moonshine training.
        """
        self.logger.info(f'Loading local dataset from {self.dataset_path}...')

        if not os.path.exists(self.dataset_path):
            self.logger.error(f'Path not found: {self.dataset_path}')
            return None

        try:
            # 1. Load the dataset from the local path provided
            dataset = load_from_disk(self.dataset_path)
            self.logger.info("Local dataset loaded into memory mapping.")

            # 2. Resample audio to 16kHz (Required for Moonshine)
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
            
            # 3. Basic Validation
            for split in ['train', 'test']:
                if split not in dataset:
                    self.logger.warning(f"Split '{split}' not found in dataset!")
                else:
                    cols = dataset[split].column_names
                    if self.text_column not in cols:
                        self.logger.error(f"Text column '{self.text_column}' not found. Available: {cols}")
                    
            return dataset

        except Exception as e:
            self.logger.error(f'Error loading dataset: {str(e)}')
            return None

    def get_cache_paths(self):
        """
        Returns a dictionary of paths for the .map() function to avoid Read-Only errors.
        """
        return {
            "train": os.path.join(self.cache_dir, "train_cache.arrow"),
            "test": os.path.join(self.cache_dir, "test_cache.arrow")
        }

# --- Example of how to use this in your training script ---
# loader = DataLoader(dataset_path="/kaggle/input/preparong-moonshine-data/bangla_moonshine_ds")
# dataset = loader.load_dataset()
# cache_files = loader.get_cache_paths()
#
# dataset = dataset.map(prepare_function, cache_file_names=cache_files, ...)
