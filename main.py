# main.py
import argparse
from pathlib import Path
from config import Config
from pipeline import ETLPipeline, TrainingPipeline
from utils import Logger

def main():
    parser = argparse.ArgumentParser(description='Movie Recommender System')
    parser.add_argument('--mode', type=str, choices=['etl', 'train', 'all'], 
                       default='all', help='Execution mode')
    parser.add_argument('--model', type=str, choices=['content', 'mf', 'ncf', 'all'],
                       default='all', help='Model type to train')
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    logger = Logger.get_logger(__name__)
    
    # Create artifacts directory
    Path('./artifacts').mkdir(exist_ok=True)
    
    logger.info(f"Starting Movie Recommender System in {args.mode} mode")
    
    try:
        # Run ETL pipeline
        if args.mode in ['etl', 'all']:
            etl_pipeline = ETLPipeline(config)
            processed_df = etl_pipeline.run()
            logger.info("ETL pipeline completed")
        
        # Run training pipeline
        if args.mode in ['train', 'all']:
            training_pipeline = TrainingPipeline(config)
            models = training_pipeline.run(model_type=args.model)
            logger.info(f"Training completed. Models trained: {list(models.keys())}")
        
        logger.info("Execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
