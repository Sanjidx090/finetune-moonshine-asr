import os
import logging
import pandas as pd

class DataLoader:
    def __init__(self, dataset_path, test_size=0.2, random_state=None):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.logger = self.setup_logging()

    def setup_logging(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        return logger

    def load_local_from_disk(self):
        self.logger.info('Loading dataset from {}...'.format(self.dataset_path))

        if not os.path.exists(self.dataset_path):
            self.logger.error('Dataset path does not exist.')
            return None

        if self.dataset_path.endswith('.csv'):
            data = pd.read_csv(self.dataset_path)
            self.logger.info('CSV dataset loaded successfully.')
        elif self.dataset_path.endswith('.json'):
            data = pd.read_json(self.dataset_path)
            self.logger.info('JSON dataset loaded successfully.')
        else:
            self.logger.error('Unsupported dataset type. Please use CSV or JSON.')
            return None

        # Split the data into train and test sets
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=self.random_state)

        self.logger.info('Dataset split into training and testing sets.')
        return train_data, test_data
