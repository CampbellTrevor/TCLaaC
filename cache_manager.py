"""
Cache management for expensive operations in TCLaaC.

This module provides intelligent caching for:
- Trained LDA models
- LOLBAS data
- Preprocessed corpora
- Analysis results

Caching significantly improves performance for iterative analysis
and reprocessing of similar datasets.
"""

import os
import hashlib
import json
import joblib
import logging
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching of expensive computations to improve performance.
    """
    
    def __init__(self, cache_dir: str = '.tclaaс_cache', max_age_days: int = 30):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
            max_age_days: Maximum age of cached items in days
        """
        self.cache_dir = Path(cache_dir)
        self.max_age_days = max_age_days
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different cache types
        self.models_dir = self.cache_dir / 'models'
        self.lolbas_dir = self.cache_dir / 'lolbas'
        self.corpus_dir = self.cache_dir / 'corpus'
        self.analysis_dir = self.cache_dir / 'analysis'
        
        for directory in [self.models_dir, self.lolbas_dir, self.corpus_dir, self.analysis_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.debug(f"Cache manager initialized at {self.cache_dir}")
    
    def _generate_key(self, data: Any) -> str:
        """
        Generate a unique cache key from data.
        
        Args:
            data: Data to generate key from (serializable)
            
        Returns:
            SHA256 hash as cache key
        """
        # Convert data to stable string representation
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """Get path for cached item."""
        type_dirs = {
            'model': self.models_dir,
            'lolbas': self.lolbas_dir,
            'corpus': self.corpus_dir,
            'analysis': self.analysis_dir
        }
        base_dir = type_dirs.get(cache_type, self.cache_dir)
        return base_dir / f"{key}.cache"
    
    def _is_valid(self, cache_path: Path) -> bool:
        """
        Check if cached item is still valid.
        
        Args:
            cache_path: Path to cached item
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False
        
        # Check age
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        max_age = datetime.now() - timedelta(days=self.max_age_days)
        
        if mtime < max_age:
            logger.debug(f"Cache expired: {cache_path.name}")
            return False
        
        return True
    
    def get(self, cache_type: str, key_data: Any) -> Optional[Any]:
        """
        Retrieve cached item.
        
        Args:
            cache_type: Type of cached item ('model', 'lolbas', 'corpus', 'analysis')
            key_data: Data to generate cache key from
            
        Returns:
            Cached data if found and valid, None otherwise
        """
        key = self._generate_key(key_data)
        cache_path = self._get_cache_path(cache_type, key)
        
        if not self._is_valid(cache_path):
            return None
        
        try:
            logger.debug(f"Cache hit: {cache_type}/{key[:8]}...")
            return joblib.load(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None
    
    def set(self, cache_type: str, key_data: Any, value: Any) -> bool:
        """
        Store item in cache.
        
        Args:
            cache_type: Type of cached item
            key_data: Data to generate cache key from
            value: Data to cache
            
        Returns:
            True if successful, False otherwise
        """
        key = self._generate_key(key_data)
        cache_path = self._get_cache_path(cache_type, key)
        
        try:
            joblib.dump(value, cache_path)
            logger.debug(f"Cache stored: {cache_type}/{key[:8]}...")
            return True
        except Exception as e:
            logger.warning(f"Failed to store cache {cache_path}: {e}")
            return False
    
    def clear(self, cache_type: Optional[str] = None) -> int:
        """
        Clear cached items.
        
        Args:
            cache_type: Type to clear (None = all types)
            
        Returns:
            Number of items cleared
        """
        count = 0
        
        if cache_type:
            # Clear specific type
            type_dir = {
                'model': self.models_dir,
                'lolbas': self.lolbas_dir,
                'corpus': self.corpus_dir,
                'analysis': self.analysis_dir
            }.get(cache_type)
            
            if type_dir:
                for cache_file in type_dir.glob('*.cache'):
                    cache_file.unlink()
                    count += 1
        else:
            # Clear all types
            for cache_file in self.cache_dir.rglob('*.cache'):
                cache_file.unlink()
                count += 1
        
        logger.info(f"Cleared {count} cached items" + (f" ({cache_type})" if cache_type else ""))
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'total_items': 0,
            'total_size_mb': 0,
            'by_type': {}
        }
        
        for cache_type, type_dir in [
            ('model', self.models_dir),
            ('lolbas', self.lolbas_dir),
            ('corpus', self.corpus_dir),
            ('analysis', self.analysis_dir)
        ]:
            items = list(type_dir.glob('*.cache'))
            size_bytes = sum(item.stat().st_size for item in items)
            
            stats['by_type'][cache_type] = {
                'items': len(items),
                'size_mb': size_bytes / (1024 * 1024)
            }
            
            stats['total_items'] += len(items)
            stats['total_size_mb'] += size_bytes / (1024 * 1024)
        
        return stats
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache items.
        
        Returns:
            Number of items removed
        """
        count = 0
        
        for cache_file in self.cache_dir.rglob('*.cache'):
            if not self._is_valid(cache_file):
                cache_file.unlink()
                count += 1
        
        if count > 0:
            logger.info(f"Cleaned up {count} expired cache items")
        
        return count


# Global cache instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
