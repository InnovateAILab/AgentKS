"""
High Energy Physics (HEP) Tools Module

MCP tools for searching scientific literature in High Energy Physics:
- INSPIRE-HEP: Comprehensive HEP literature database
- CERN Document Server (CDS): CERN's institutional repository
- arXiv: Preprint server for physics and mathematics

These tools are registered with the main FastMCP server instance.
"""
import os
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


def register_hep_tools(mcp):
    """
    Register all HEP tools with the provided FastMCP server instance.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.tool()
    def inspirehep_search(query: str, max_results: int = 10, sort: str = "mostrecent") -> Dict[str, Any]:
        """
        Search INSPIRE-HEP literature database for High Energy Physics papers.
        INSPIRE-HEP is the most comprehensive database for HEP literature.
        
        Args:
            query: Search query (supports INSPIRE search syntax)
            max_results: Maximum number of results to return (default: 10)
            sort: Sort order - "mostrecent", "mostcited" (default: "mostrecent")
        
        Returns:
            Dictionary with search results including titles, authors, citations, and URLs
        """
        base_url = os.getenv("INSPIRE_BASE_URL", "https://inspirehep.net")
        
        try:
            params = {
                "q": query,
                "size": str(max_results),
                "sort": sort
            }
            
            response = requests.get(
                f"{base_url.rstrip('/')}/api/literature",
                params=params,
                headers={"Accept": "application/json"},
                timeout=25
            )
            response.raise_for_status()
            data = response.json()
            
            hits = ((data or {}).get("hits") or {}).get("hits") or []
            results = []
            
            for hit in hits[:max_results]:
                metadata = hit.get("metadata") or {}
                
                # Extract title
                titles = metadata.get("titles") or []
                title = titles[0].get("title") if titles and isinstance(titles[0], dict) else None
                
                # Extract authors
                authors_data = metadata.get("authors") or []
                authors = [a.get("full_name") for a in authors_data[:3] if isinstance(a, dict)]
                if len(authors_data) > 3:
                    authors.append(f"... and {len(authors_data) - 3} more")
                
                # Extract citation count
                citation_count = metadata.get("citation_count", 0)
                
                # Extract publication info
                publication_info = metadata.get("publication_info") or []
                journal = publication_info[0].get("journal_title") if publication_info else None
                
                # Extract URLs
                links = hit.get("links") or {}
                url = links.get("html") or links.get("json")
                
                # Extract arXiv ID if available
                arxiv_eprints = metadata.get("arxiv_eprints") or []
                arxiv_id = arxiv_eprints[0].get("value") if arxiv_eprints else None
                
                results.append({
                    "title": title,
                    "authors": authors,
                    "citation_count": citation_count,
                    "journal": journal,
                    "arxiv_id": arxiv_id,
                    "url": url
                })
            
            return {
                "query": query,
                "total": len(results),
                "results": results,
                "source": "inspirehep",
                "base_url": base_url
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"INSPIRE-HEP search error: {e}")
            return {
                "error": f"INSPIRE-HEP API error: {str(e)}",
                "query": query,
                "source": "inspirehep"
            }
        except Exception as e:
            logger.error(f"Unexpected INSPIRE-HEP error: {e}")
            return {
                "error": str(e),
                "query": query,
                "source": "inspirehep"
            }
    
    
    @mcp.tool()
    def cds_search(query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search CERN Document Server (CDS) for documents, preprints, and multimedia.
        CDS is CERN's institutional repository containing theses, reports, and publications.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 10)
        
        Returns:
            Dictionary with search results including record IDs, titles, and URLs
        """
        base_url = os.getenv("CDS_BASE_URL", "https://cds.cern.ch")
        
        try:
            params = {
                "p": query,
                "of": "recjson",  # JSON format
                "rg": str(max_results),
                "jrec": "1",
                "ot": "recid,creation_date,authors[0],number_of_authors,title,abstract"
            }
            
            response = requests.get(
                f"{base_url.rstrip('/')}/search",
                params=params,
                timeout=25
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for record in (data or [])[:max_results]:
                recid = record.get("recid")
                
                # Extract title (can be dict or string)
                title = None
                t = record.get("title")
                if isinstance(t, dict):
                    title = t.get("title") or t.get("subtitle")
                elif isinstance(t, str):
                    title = t
                
                # Extract authors
                authors_data = record.get("authors") or []
                if isinstance(authors_data, list):
                    authors = [a.get("name") if isinstance(a, dict) else str(a) for a in authors_data[:3]]
                    if record.get("number_of_authors", 0) > 3:
                        authors.append(f"... and {record.get('number_of_authors') - 3} more")
                else:
                    authors = []
                
                # Extract abstract
                abstract = record.get("abstract")
                if abstract and len(abstract) > 300:
                    abstract = abstract[:300] + "..."
                
                # Build record URL
                record_url = f"{base_url.rstrip('/')}/record/{recid}" if recid else None
                
                results.append({
                    "recid": recid,
                    "title": title,
                    "authors": authors,
                    "creation_date": record.get("creation_date"),
                    "abstract": abstract,
                    "url": record_url
                })
            
            return {
                "query": query,
                "total": len(results),
                "results": results,
                "source": "cds",
                "base_url": base_url
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CDS search error: {e}")
            return {
                "error": f"CDS API error: {str(e)}",
                "query": query,
                "source": "cds"
            }
        except Exception as e:
            logger.error(f"Unexpected CDS error: {e}")
            return {
                "error": str(e),
                "query": query,
                "source": "cds"
            }
    
    
    @mcp.tool()
    def arxiv_hep_search(
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        category: str = "hep-ph"
    ) -> Dict[str, Any]:
        """
        Search arXiv specifically for High Energy Physics papers.
        This is a specialized wrapper around arXiv API with HEP-specific defaults.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (default: 10)
            sort_by: Sort criteria - "relevance", "lastUpdatedDate", "submittedDate" (default: "relevance")
            sort_order: "ascending" or "descending" (default: "descending")
            category: HEP category - "hep-ph" (phenomenology), "hep-th" (theory), 
                     "hep-ex" (experiment), "hep-lat" (lattice) (default: "hep-ph")
        
        Returns:
            Dictionary with paper information including titles, authors, abstracts, and PDF URLs
        """
        api_url = os.getenv("ARXIV_API_URL", "http://export.arxiv.org/api/query")
        
        try:
            # Build search query with category filter
            search_query = f"cat:{category} AND all:{query}" if category else f"all:{query}"
            
            params = {
                "search_query": search_query,
                "start": "0",
                "max_results": str(max_results),
                "sortBy": sort_by,
                "sortOrder": sort_order
            }
            
            response = requests.get(api_url, params=params, timeout=25)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            
            results = []
            for entry in root.findall("a:entry", ns):
                # Extract title
                title_el = entry.find("a:title", ns)
                title = (title_el.text or "").strip() if title_el is not None else None
                
                # Extract authors
                author_els = entry.findall("a:author/a:name", ns)
                authors = [(a.text or "").strip() for a in author_els[:3]]
                if len(author_els) > 3:
                    authors.append(f"... and {len(author_els) - 3} more")
                
                # Extract summary/abstract
                summary_el = entry.find("a:summary", ns)
                summary = (summary_el.text or "").strip() if summary_el is not None else None
                if summary and len(summary) > 400:
                    summary = summary[:400] + "..."
                
                # Extract links
                link_el = entry.find("a:link[@rel='alternate']", ns)
                url = link_el.attrib.get("href") if link_el is not None else None
                
                # Extract PDF link
                pdf_link_el = entry.find("a:link[@title='pdf']", ns)
                pdf_url = pdf_link_el.attrib.get("href") if pdf_link_el is not None else None
                
                # Extract arXiv ID from URL
                arxiv_id = None
                if url:
                    arxiv_id = url.split("/")[-1] if "/" in url else None
                
                # Extract published date
                published_el = entry.find("a:published", ns)
                published = (published_el.text or "").strip() if published_el is not None else None
                
                # Extract categories
                category_els = entry.findall("a:category", ns)
                categories = [c.attrib.get("term") for c in category_els if c.attrib.get("term")]
                
                results.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "abstract": summary,
                    "published": published,
                    "categories": categories,
                    "url": url,
                    "pdf_url": pdf_url
                })
            
            return {
                "query": query,
                "category": category,
                "total": len(results),
                "results": results,
                "source": "arxiv",
                "sort_by": sort_by,
                "sort_order": sort_order
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"arXiv HEP search error: {e}")
            return {
                "error": f"arXiv API error: {str(e)}",
                "query": query,
                "source": "arxiv"
            }
        except ET.ParseError as e:
            logger.error(f"arXiv XML parse error: {e}")
            return {
                "error": f"Failed to parse arXiv response: {str(e)}",
                "query": query,
                "source": "arxiv"
            }
        except Exception as e:
            logger.error(f"Unexpected arXiv error: {e}")
            return {
                "error": str(e),
                "query": query,
                "source": "arxiv"
            }
    
    logger.info("Registered 3 HEP tools: inspirehep_search, cds_search, arxiv_hep_search")
