"""
URL Discovery Utilities

Discover and extract related URLs from web pages.
Useful for documentation sites, blogs, and multi-page content.
"""

import re
import logging
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse, urlunparse
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class URLDiscoverer:
    """Discover related URLs from a parent URL."""
    
    def __init__(self, max_depth: int = 2, max_urls: int = 100, timeout: int = 30):
        """
        Initialize URL discoverer.
        
        Args:
            max_depth: Maximum depth to crawl (0 = just parent page)
            max_urls: Maximum URLs to discover
            timeout: HTTP request timeout in seconds
        """
        self.max_depth = max_depth
        self.max_urls = max_urls
        self.timeout = timeout
        self.user_agent = 'AgentKS-URLDiscoverer/1.0'
    
    def discover_urls(self, parent_url: str, same_domain_only: bool = True) -> List[Dict[str, str]]:
        """
        Discover related URLs from parent URL.
        
        Args:
            parent_url: The starting URL
            same_domain_only: Only discover URLs from same domain
            
        Returns:
            List of dicts with 'url', 'title', 'depth' keys
        """
        discovered = []
        visited = set()
        to_visit = [(parent_url, 0)]  # (url, depth)
        parent_domain = self._get_domain(parent_url)
        
        while to_visit and len(discovered) < self.max_urls:
            current_url, depth = to_visit.pop(0)
            
            # Skip if already visited or max depth reached
            if current_url in visited or depth > self.max_depth:
                continue
            
            visited.add(current_url)
            
            try:
                # Fetch page
                response = requests.get(
                    current_url,
                    timeout=self.timeout,
                    headers={'User-Agent': self.user_agent},
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get page title
                title = self._extract_title(soup, current_url)
                
                # Add current URL to discovered list
                discovered.append({
                    'url': current_url,
                    'title': title,
                    'depth': depth,
                    'selected': depth == 0  # Auto-select parent URL
                })
                
                # Find all links if not at max depth
                if depth < self.max_depth:
                    links = self._extract_links(soup, current_url, parent_domain, same_domain_only)
                    
                    # Add new links to visit queue
                    for link in links:
                        if link not in visited and len(discovered) < self.max_urls:
                            to_visit.append((link, depth + 1))
                
            except Exception as e:
                logger.warning(f"Failed to fetch {current_url}: {e}")
                continue
        
        logger.info(f"Discovered {len(discovered)} URLs from {parent_url}")
        return discovered
    
    def discover_documentation_urls(self, parent_url: str) -> List[Dict[str, str]]:
        """
        Specialized discovery for documentation sites (ReadTheDocs, MkDocs, etc.).
        
        Args:
            parent_url: Documentation site URL
            
        Returns:
            List of discovered documentation pages
        """
        discovered = []
        
        try:
            response = requests.get(
                parent_url,
                timeout=self.timeout,
                headers={'User-Agent': self.user_agent}
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Common documentation navigation selectors
            nav_selectors = [
                'nav.toctree',  # Sphinx/ReadTheDocs
                'nav.md-nav',   # MkDocs Material
                '.toctree-wrapper',
                '.sidebar-nav',
                'nav[role="navigation"]',
                '.documentation-nav'
            ]
            
            links = set()
            
            # Try each selector
            for selector in nav_selectors:
                nav_element = soup.select_one(selector)
                if nav_element:
                    for link in nav_element.find_all('a', href=True):
                        href = link.get('href')
                        if href:
                            absolute_url = urljoin(parent_url, href)
                            # Remove anchors
                            absolute_url = absolute_url.split('#')[0]
                            links.add(absolute_url)
            
            # If no navigation found, fall back to all links
            if not links:
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and not href.startswith('#'):
                        absolute_url = urljoin(parent_url, href)
                        absolute_url = absolute_url.split('#')[0]
                        if self._is_same_domain(parent_url, absolute_url):
                            links.add(absolute_url)
            
            # Convert to list of dicts
            for i, url in enumerate(sorted(links)[:self.max_urls]):
                discovered.append({
                    'url': url,
                    'title': self._extract_title_from_url(url),
                    'depth': 1,
                    'selected': True  # Auto-select all documentation pages
                })
            
            logger.info(f"Discovered {len(discovered)} documentation URLs")
            
        except Exception as e:
            logger.error(f"Failed to discover documentation URLs: {e}")
        
        return discovered
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, parent_domain: str, 
                      same_domain_only: bool) -> List[str]:
        """Extract and normalize links from page."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            
            # Skip anchors, mailto, javascript, etc.
            if not href or href.startswith(('#', 'mailto:', 'javascript:', 'tel:')):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Remove fragment
            absolute_url = absolute_url.split('#')[0]
            
            # Skip if different domain and same_domain_only is True
            if same_domain_only and not self._is_same_domain(base_url, absolute_url):
                continue
            
            # Skip common file types we don't want
            if self._is_excluded_file(absolute_url):
                continue
            
            links.append(absolute_url)
        
        return list(set(links))  # Remove duplicates
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title."""
        # Try <title> tag
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try <h1> tag
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        # Fall back to URL
        return self._extract_title_from_url(url)
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a reasonable title from URL path."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if not path:
            return parsed.netloc
        
        # Get last path component
        last_part = path.split('/')[-1]
        
        # Remove file extension
        if '.' in last_part:
            last_part = last_part.rsplit('.', 1)[0]
        
        # Replace hyphens/underscores with spaces and title case
        title = last_part.replace('-', ' ').replace('_', ' ').title()
        
        return title or url
    
    def _get_domain(self, url: str) -> str:
        """Get domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc
    
    def _is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs have the same domain."""
        return self._get_domain(url1) == self._get_domain(url2)
    
    def _is_excluded_file(self, url: str) -> bool:
        """Check if URL points to excluded file type."""
        excluded_extensions = {
            '.pdf', '.zip', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif',
            '.mp4', '.mp3', '.avi', '.mov', '.exe', '.dmg', '.iso'
        }
        
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        return any(path.endswith(ext) for ext in excluded_extensions)


def discover_urls_quick(url: str, max_urls: int = 50) -> List[Dict[str, str]]:
    """
    Quick URL discovery helper function.
    
    Args:
        url: Parent URL to discover from
        max_urls: Maximum URLs to discover
        
    Returns:
        List of discovered URLs with metadata
    """
    discoverer = URLDiscoverer(max_depth=1, max_urls=max_urls)
    
    # Check if it looks like documentation
    if any(pattern in url.lower() for pattern in ['docs', 'documentation', 'readthedocs']):
        return discoverer.discover_documentation_urls(url)
    else:
        return discoverer.discover_urls(url, same_domain_only=True)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with ReadTheDocs
    urls = discover_urls_quick("https://panda-wms.readthedocs.io/en/latest/", max_urls=20)
    
    print(f"\nDiscovered {len(urls)} URLs:")
    for i, url_info in enumerate(urls[:10], 1):
        print(f"{i}. [{url_info['depth']}] {url_info['title']}")
        print(f"   {url_info['url']}")
