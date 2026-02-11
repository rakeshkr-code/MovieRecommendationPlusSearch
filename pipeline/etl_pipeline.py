# pipeline/etl_pipeline.py
import pandas as pd
from typing import Tuple
# from ..data_scripts import DataLoader, DataProcessor, DatabaseManager
from data_scripts import DataLoader, DataProcessor, DatabaseManager
# from ..utils import Logger
from utils import Logger

class ETLPipeline:
    """ETL Pipeline for data processing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = Logger.get_logger(__name__)
        self.data_loader = DataLoader(
            config.data.raw_movies_path,
            config.data.raw_credits_path
        )
        self.data_processor = DataProcessor()
        self.db_manager = DatabaseManager(config.data.db_path)
    
    def extract(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract data from CSV files"""
        self.logger.info("Extracting data from CSV files...")
        movies_df, credits_df = self.data_loader.load_raw_data()
        self.data_loader.validate_data(movies_df, credits_df)
        self.logger.info(f"Extracted {len(movies_df)} movies and {len(credits_df)} credits")
        return movies_df, credits_df
    
    def transform(self, movies_df: pd.DataFrame, credits_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform data into both processed and metadata tables"""
        self.logger.info("Transforming data...")
        
        # Create processed_movies table (for content-based recommendations)
        processed_df = self.data_processor.process_dataframe(movies_df, credits_df)
        self.logger.info(f"Processed {len(processed_df)} movies for recommendations table")
        
        # Create movies_metadata table (for SQL queries)
        metadata_df = self.data_processor.process_metadata_dataframe(movies_df, credits_df)
        self.logger.info(f"Processed {len(metadata_df)} movies for metadata table")
        
        return processed_df, metadata_df
    
    def load(self, movies_df: pd.DataFrame, credits_df: pd.DataFrame, processed_df: pd.DataFrame, metadata_df: pd.DataFrame):
        """Load data into database"""
        self.logger.info("Loading data into database...")
        
        with self.db_manager as db:
            # Save raw data
            db.save_dataframe(movies_df, self.config.data.raw_movies_table)
            db.save_dataframe(credits_df, self.config.data.raw_credits_table)
            
            # Save processed data (for recommendations)
            db.save_dataframe(processed_df, self.config.data.processed_table)
            
            # Save metadata (for SQL queries)
            db.save_dataframe(metadata_df, 'movies_metadata')
        
        self.logger.info("Data loaded successfully into database")
        self.logger.info(f"  - {self.config.data.processed_table}: {len(processed_df)} rows (for recommendations)")
        self.logger.info(f"  - movies_metadata: {len(metadata_df)} rows (for SQL queries)")
    
    def run(self) -> pd.DataFrame:
        """Run the complete ETL pipeline"""
        self.logger.info("Starting ETL pipeline...")
        
        # Extract
        movies_df, credits_df = self.extract()
        
        # Transform - now returns both tables
        processed_df, metadata_df = self.transform(movies_df, credits_df)
        
        # Load - saves both tables
        self.load(movies_df, credits_df, processed_df, metadata_df)
        
        self.logger.info("ETL pipeline completed successfully")
        return processed_df
