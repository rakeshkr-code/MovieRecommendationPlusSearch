# utils/tmdb_api.py
"""
TMDB API Integration for fetching movie posters, ratings, and details
"""

import requests
from typing import Optional, Dict, Any
import logging
from functools import lru_cache

class TMDBClient:
    """Client for interacting with The Movie Database (TMDB) API"""
    
    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TMDB client
        
        Args:
            api_key: TMDB API key (v3 auth)
        """
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        if not api_key:
            self.logger.warning("TMDB API key not provided. Some features will be unavailable.")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make API request to TMDB
        
        Args:
            endpoint: API endpoint (e.g., '/movie/550')
            params: Query parameters
            
        Returns:
            JSON response or None if error
        """
        if not self.api_key:
            return None
        
        url = f"{self.BASE_URL}{endpoint}"
        
        # Add API key to params
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"TMDB API request failed: {e}")
            return None
    
    @lru_cache(maxsize=1000)
    def get_movie_details(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a movie
        
        Args:
            movie_id: TMDB movie ID
            
        Returns:
            Dictionary with movie details or None
        """
        data = self._make_request(f"/movie/{movie_id}")
        
        if not data:
            return None
        
        return {
            'id': data.get('id'),
            'title': data.get('title'),
            'original_title': data.get('original_title'),
            'overview': data.get('overview', 'No overview available.'),
            'poster_path': data.get('poster_path'),
            'backdrop_path': data.get('backdrop_path'),
            'vote_average': data.get('vote_average', 0),
            'vote_count': data.get('vote_count', 0),
            'release_date': data.get('release_date', 'Unknown'),
            'genres': [g['name'] for g in data.get('genres', [])],
            'runtime': data.get('runtime'),
            'tagline': data.get('tagline', ''),
            'popularity': data.get('popularity', 0)
        }
    
    @lru_cache(maxsize=500)
    def search_movie(self, query: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Search for a movie by title
        
        Args:
            query: Movie title to search
            year: Release year (optional, for better matching)
            
        Returns:
            First matching movie details or None
        """
        params = {'query': query}
        if year:
            params['year'] = year
        
        data = self._make_request("/search/movie", params)
        
        if not data or not data.get('results'):
            return None
        
        # Return first result
        first_result = data['results'][0]
        return {
            'id': first_result.get('id'),
            'title': first_result.get('title'),
            'release_date': first_result.get('release_date', 'Unknown'),
            'poster_path': first_result.get('poster_path'),
            'vote_average': first_result.get('vote_average', 0)
        }
    
    def get_poster_url(self, poster_path: Optional[str], size: str = "w500") -> Optional[str]:
        """
        Construct full poster image URL
        
        Args:
            poster_path: Poster path from TMDB API
            size: Image size (w92, w154, w185, w342, w500, w780, original)
            
        Returns:
            Full poster URL or None
        """
        if not poster_path:
            return None
        
        return f"{self.IMAGE_BASE_URL}/{size}{poster_path}"
    
    def get_backdrop_url(self, backdrop_path: Optional[str], size: str = "w1280") -> Optional[str]:
        """
        Construct full backdrop image URL
        
        Args:
            backdrop_path: Backdrop path from TMDB API
            size: Image size (w300, w780, w1280, original)
            
        Returns:
            Full backdrop URL or None
        """
        if not backdrop_path:
            return None
        
        return f"{self.IMAGE_BASE_URL}/{size}{backdrop_path}"
    
    def get_movie_with_details(self, movie_id: int) -> Dict[str, Any]:
        """
        Get movie details with constructed image URLs
        
        Args:
            movie_id: TMDB movie ID
            
        Returns:
            Dictionary with details and full image URLs
        """
        details = self.get_movie_details(movie_id)
        
        if not details:
            return {
                'id': movie_id,
                'title': 'Unknown',
                'overview': 'Details not available',
                'poster_url': None,
                'vote_average': 0,
                'genres': [],
                'release_date': 'Unknown'
            }
        
        # Add full URLs
        details['poster_url'] = self.get_poster_url(details.get('poster_path'))
        details['backdrop_url'] = self.get_backdrop_url(details.get('backdrop_path'))
        
        return details


def get_tmdb_client(api_key: Optional[str] = None) -> TMDBClient:
    """
    Factory function to create TMDB client
    
    Args:
        api_key: TMDB API key
        
    Returns:
        TMDBClient instance
    """
    return TMDBClient(api_key=api_key)
