# inference.py
"""
Movie Recommendation Inference Script

Usage:
    python inference.py --model content --movie "Avatar" --top-n 5
    python inference.py --model mf --user 42 --top-n 10
    python inference.py  # Interactive mode
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

from config import Config
from data_scripts import DatabaseManager
from models import ContentBasedRecommender, MFRecommender, NCFRecommender
from utils import Logger
from utils.movie_search import MovieSearchEngine


class InferenceEngine:
    """Engine for generating movie recommendations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger.get_logger(__name__)
        self.db_manager = DatabaseManager(config.data.db_path)
        
        # Model storage
        self.models = {
            'content': None,
            'mf': None,
            'ncf': None
        }
        
        # Data storage
        self.movies_df = None
        self.search_engine = None
        
        # Load movie data
        self._load_movie_data()
    
    def _load_movie_data(self):
        """Load movie data from database"""
        self.logger.info("Loading movie data from database...")
        try:
            with self.db_manager as db:
                self.movies_df = db.load_dataframe(self.config.data.processed_table)
            
            self.search_engine = MovieSearchEngine(self.movies_df)
            self.logger.info(f"Loaded {len(self.movies_df)} movies")
        except Exception as e:
            self.logger.error(f"Failed to load movie data: {e}")
            raise
    
    def load_model(self, model_type: str) -> bool:
        """
        Load trained model from artifacts
        
        Args:
            model_type: 'content', 'mf', or 'ncf'
            
        Returns:
            True if loaded successfully
        """
        model_paths = {
            'content': './artifacts/content_based_model.pkl',
            'mf': './artifacts/mf_model.pkl',
            'ncf': './artifacts/ncf_model.pkl'
        }
        
        if model_type not in model_paths:
            self.logger.error(f"Invalid model type: {model_type}")
            return False
        
        model_path = Path(model_paths[model_type])
        
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            self.logger.info(f"Please train the model first: python main.py --mode train --model {model_type}")
            return False
        
        try:
            self.logger.info(f"Loading {model_type} model from {model_path}...")
            
            if model_type == 'content':
                self.models[model_type] = ContentBasedRecommender.load_model(str(model_path))
            elif model_type == 'mf':
                self.models[model_type] = MFRecommender.load_model(str(model_path))
            elif model_type == 'ncf':
                self.models[model_type] = NCFRecommender.load_model(str(model_path))
            
            self.logger.info(f"Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def recommend_by_movie(self, movie_name: str, model_type: str = 'content', 
                          top_n: int = 10, interactive: bool = True) -> Optional[List[Tuple[int, float]]]:
        """
        Get movie recommendations based on a movie name
        
        Args:
            movie_name: Name of the movie
            model_type: Type of model to use ('content', 'mf', 'ncf')
            top_n: Number of recommendations
            interactive: Whether to use interactive mode for disambiguation
            
        Returns:
            List of (movie_id, score) tuples or None
        """
        # Only content-based model supports movie-to-movie recommendations
        if model_type in ['mf', 'ncf']:
            print(f"\n⚠️  Warning: {model_type.upper()} model is designed for user-based recommendations.")
            print("For movie-to-movie similarity, please use 'content' model.")
            print("Falling back to Content-Based model...\n")
            model_type = 'content'
        
        # Load model if not already loaded
        if self.models[model_type] is None:
            if not self.load_model(model_type):
                return None
        
        # Find movie
        result = self.search_engine.find_movie(movie_name, interactive=interactive)
        
        if result is None:
            print(f"\n❌ Movie '{movie_name}' not found in database.")
            print("\nSuggestions:")
            print("  • Check spelling")
            print("  • Try shorter/partial title")
            print("  • Use --interactive flag for multiple options")
            return None
        
        movie_idx, movie_id, movie_title = result
        
        print(f"\n✓ Found: {movie_title}")
        print("─" * 70)
        
        # Get recommendations
        try:
            model = self.models[model_type]
            recommendations = model.recommend(movie_idx, top_n=top_n)
            return recommendations, movie_title
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            print(f"\n❌ Error generating recommendations: {e}")
            return None
    
    def recommend_by_user(self, user_id: int, model_type: str = 'mf', 
                         top_n: int = 10) -> Optional[List[Tuple[int, float]]]:
        """
        Get movie recommendations for a user
        
        Args:
            user_id: User ID
            model_type: Type of model to use ('mf' or 'ncf')
            top_n: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples or None
        """
        if model_type == 'content':
            print("\n⚠️  Content-Based model doesn't support user-based recommendations.")
            print("Please use 'mf' or 'ncf' model with --movie option instead.")
            return None
        
        # Load model if not already loaded
        if self.models[model_type] is None:
            if not self.load_model(model_type):
                return None
        
        try:
            model = self.models[model_type]
            recommendations = model.recommend(user_id, top_n=top_n)
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            print(f"\n❌ Error generating recommendations: {e}")
            return None
    
    def display_recommendations(self, recommendations: List[Tuple[int, float]], 
                               input_title: str, model_type: str):
        """
        Display recommendations in a pretty format
        
        Args:
            recommendations: List of (movie_id, score) tuples
            input_title: Input movie title
            model_type: Model type used
        """
        model_names = {
            'content': 'Content-Based Filtering',
            'mf': 'Matrix Factorization',
            'ncf': 'Neural Collaborative Filtering'
        }
        
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " Movie Recommendation System - Inference ".center(68) + "║")
        print("╚" + "═" * 68 + "╝")
        
        print(f"\n📽️  Input Movie: {input_title}")
        print(f"🤖 Model: {model_names.get(model_type, model_type)}")
        print(f"📊 Top {len(recommendations)} Recommendations:")
        print("\n" + "─" * 70)
        
        for i, (movie_id, score) in enumerate(recommendations, 1):
            # Get movie title by ID
            movie_info = self.search_engine.get_movie_by_id(movie_id)
            if movie_info:
                _, _, title = movie_info
            else:
                title = f"Movie ID: {movie_id}"
            
            # Format score based on model type
            if model_type == 'content':
                score_str = f"Similarity: {score:.3f}"
            else:
                score_str = f"Score: {score:.3f}"
            
            print(f"{i:2d}. {title:<50s} ({score_str})")
        
        print("─" * 70 + "\n")
    
    def display_user_recommendations(self, recommendations: List[Tuple[int, float]], 
                                    user_id: int, model_type: str):
        """
        Display user recommendations in a pretty format
        
        Args:
            recommendations: List of (movie_id, score) tuples
            user_id: User ID
            model_type: Model type used
        """
        model_names = {
            'mf': 'Matrix Factorization',
            'ncf': 'Neural Collaborative Filtering'
        }
        
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " Movie Recommendation System - Inference ".center(68) + "║")
        print("╚" + "═" * 68 + "╝")
        
        print(f"\n👤 User ID: {user_id}")
        print(f"🤖 Model: {model_names.get(model_type, model_type)}")
        print(f"📊 Top {len(recommendations)} Recommendations:")
        print("\n" + "─" * 70)
        
        for i, (movie_id, score) in enumerate(recommendations, 1):
            # Get movie title by ID
            movie_info = self.search_engine.get_movie_by_id(movie_id)
            if movie_info:
                _, _, title = movie_info
            else:
                title = f"Movie ID: {movie_id}"
            
            print(f"{i:2d}. {title:<50s} (Score: {score:.3f})")
        
        print("─" * 70 + "\n")


def interactive_mode():
    """Run inference in interactive mode"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " Movie Recommendation System - Interactive Mode ".center(68) + "║")
    print("╚" + "═" * 68 + "╝\n")
    
    config = Config()
    engine = InferenceEngine(config)
    
    # Select recommendation type
    print("Select recommendation type:")
    print("  1. Movie-to-Movie similarity (Content-Based)")
    print("  2. User-based recommendations (Matrix Factorization)")
    print("  3. User-based recommendations (Neural CF)")
    print()
    
    while True:
        try:
            choice = input("Enter choice (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            return
    
    if choice == '1':
        # Movie-to-movie
        movie_name = input("\nEnter movie name: ").strip()
        top_n = input("Number of recommendations (default 10): ").strip()
        top_n = int(top_n) if top_n else 10
        
        result = engine.recommend_by_movie(movie_name, 'content', top_n, interactive=True)
        
        if result:
            recommendations, input_title = result
            engine.display_recommendations(recommendations, input_title, 'content')
    
    else:
        # User-based
        model_type = 'mf' if choice == '2' else 'ncf'
        user_id = input("\nEnter user ID: ").strip()
        
        try:
            user_id = int(user_id)
        except ValueError:
            print("❌ Invalid user ID")
            return
        
        top_n = input("Number of recommendations (default 10): ").strip()
        top_n = int(top_n) if top_n else 10
        
        recommendations = engine.recommend_by_user(user_id, model_type, top_n)
        
        if recommendations:
            engine.display_user_recommendations(recommendations, user_id, model_type)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Movie Recommendation System - Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Movie-to-movie recommendations
  python inference.py --model content --movie "Avatar" --top-n 5
  
  # User-based recommendations
  python inference.py --model mf --user 42 --top-n 10
  
  # Interactive mode
  python inference.py
        """
    )
    
    parser.add_argument('--model', type=str, choices=['content', 'mf', 'ncf'],
                       help='Model type to use')
    parser.add_argument('--movie', type=str,
                       help='Movie name for similarity-based recommendations')
    parser.add_argument('--user', type=int,
                       help='User ID for user-based recommendations')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of recommendations (default: 10)')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive mode for movie selection')
    
    args = parser.parse_args()
    
    # Interactive mode if no arguments
    if not args.model:
        interactive_mode()
        return
    
    # Initialize
    config = Config()
    engine = InferenceEngine(config)
    
    # Movie-based or user-based?
    if args.movie:
        # Movie-to-movie recommendations
        result = engine.recommend_by_movie(
            args.movie, 
            args.model, 
            args.top_n,
            interactive=args.interactive
        )
        
        if result:
            recommendations, input_title = result
            engine.display_recommendations(recommendations, input_title, args.model)
    
    elif args.user:
        # User-based recommendations
        recommendations = engine.recommend_by_user(args.user, args.model, args.top_n)
        
        if recommendations:
            engine.display_user_recommendations(recommendations, args.user, args.model)
    
    else:
        print("❌ Error: Please specify either --movie or --user")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
