# utils/movie_search.py
import pandas as pd
from difflib import get_close_matches
from typing import List, Tuple, Optional

class MovieSearchEngine:
    """Search engine for finding movies by name with fuzzy matching"""
    
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize search engine with movie data
        
        Args:
            movies_df: DataFrame with 'id' and 'title' columns
        """
        self.df = movies_df
        self.titles = movies_df['title'].tolist()
        self.title_to_idx = {title: idx for idx, title in enumerate(movies_df['title'])}
        self.title_to_id = {title: movie_id for title, movie_id in 
                           zip(movies_df['title'], movies_df['id'])}
    
    def find_exact_match(self, query: str) -> Optional[Tuple[int, int, str]]:
        """
        Find exact match for movie title
        
        Args:
            query: Movie title to search for
            
        Returns:
            Tuple of (movie_index, movie_id, movie_title) or None
        """
        # Case-insensitive exact match
        for title in self.titles:
            if title.lower() == query.lower():
                idx = self.title_to_idx[title]
                movie_id = self.title_to_id[title]
                return (idx, movie_id, title)
        return None
    
    def find_closest_match(self, query: str, n: int = 1, cutoff: float = 0.6) -> List[Tuple[int, int, str, float]]:
        """
        Find closest matching movie titles using fuzzy matching
        
        Args:
            query: Movie title to search for
            n: Number of matches to return
            cutoff: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of tuples: (movie_index, movie_id, movie_title, similarity_score)
        """
        matches = get_close_matches(query, self.titles, n=n, cutoff=cutoff)
        
        results = []
        for title in matches:
            idx = self.title_to_idx[title]
            movie_id = self.title_to_id[title]
            # Calculate rough similarity score
            similarity = self._calculate_similarity(query.lower(), title.lower())
            results.append((idx, movie_id, title, similarity))
        
        return results
    
    def search_movies(self, query: str, max_results: int = 5) -> List[Tuple[int, int, str]]:
        """
        Search for movies with partial matching
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of tuples: (movie_index, movie_id, movie_title)
        """
        query_lower = query.lower()
        results = []
        
        for title in self.titles:
            if query_lower in title.lower():
                idx = self.title_to_idx[title]
                movie_id = self.title_to_id[title]
                results.append((idx, movie_id, title))
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def find_movie(self, query: str, interactive: bool = True) -> Optional[Tuple[int, int, str]]:
        """
        Smart movie finder with fallback strategies
        
        Args:
            query: Movie title query
            interactive: If True, ask user to select from multiple matches
            
        Returns:
            Tuple of (movie_index, movie_id, movie_title) or None
        """
        # Try exact match first
        exact = self.find_exact_match(query)
        if exact:
            return exact
        
        # Try fuzzy matching
        fuzzy_matches = self.find_closest_match(query, n=5, cutoff=0.5)
        
        if not fuzzy_matches:
            # Try partial matching
            partial_matches = self.search_movies(query, max_results=5)
            if not partial_matches:
                return None
            
            if len(partial_matches) == 1:
                return partial_matches[0]
            
            if interactive:
                return self._select_from_matches(partial_matches, query)
            else:
                return partial_matches[0]
        
        if len(fuzzy_matches) == 1:
            return fuzzy_matches[0][:3]  # Return without similarity score
        
        if interactive:
            # Convert fuzzy matches format to regular format
            matches = [(idx, movie_id, title) for idx, movie_id, title, _ in fuzzy_matches]
            return self._select_from_matches(matches, query)
        else:
            # Return best match
            return fuzzy_matches[0][:3]
    
    def _select_from_matches(self, matches: List[Tuple[int, int, str]], query: str) -> Optional[Tuple[int, int, str]]:
        """
        Interactive selection from multiple matches
        
        Args:
            matches: List of (movie_index, movie_id, movie_title) tuples
            query: Original query string
            
        Returns:
            Selected match or None
        """
        print(f"\nMultiple movies found matching '{query}':")
        print("─" * 60)
        for i, (_, _, title) in enumerate(matches, 1):
            print(f"{i}. {title}")
        print("0. None of the above")
        print("─" * 60)
        
        while True:
            try:
                choice = input("\nSelect a movie (enter number): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    return None
                elif 1 <= choice_num <= len(matches):
                    return matches[choice_num - 1]
                else:
                    print(f"Please enter a number between 0 and {len(matches)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nSearch cancelled.")
                return None
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate simple similarity score between two strings
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Use Levenshtein-like approach (simplified)
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()
    
    def get_movie_by_id(self, movie_id: int) -> Optional[Tuple[int, int, str]]:
        """
        Get movie by ID
        
        Args:
            movie_id: Movie ID from database
            
        Returns:
            Tuple of (movie_index, movie_id, movie_title) or None
        """
        for idx, (title, mid) in enumerate(zip(self.df['title'], self.df['id'])):
            if mid == movie_id:
                return (idx, mid, title)
        return None
