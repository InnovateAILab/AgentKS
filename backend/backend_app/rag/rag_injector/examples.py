"""
RAG Injection Service - Usage Examples

This file contains practical examples of using the RAG injection REST API.
Run these examples after the service is started on http://localhost:4002
"""

import requests
import json
from typing import List, Dict, Any

# Base URL for the service
BASE_URL = "http://localhost:4002"

def print_response(response: requests.Response):
    """Pretty print HTTP response."""
    print(f"\nStatus: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)


# =========================
# Example 1: Service Health Check
# =========================
def example_health_check():
    """Check if the service is running and healthy."""
    print("=" * 60)
    print("Example 1: Health Check")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)


# =========================
# Example 2: Quick Inject (Easiest Method)
# =========================
def example_quick_inject():
    """Quick inject - automatically creates group if needed."""
    print("\n" + "=" * 60)
    print("Example 2: Quick Inject (Easiest Method)")
    print("=" * 60)
    
    # Example 1: Minimal request (uses defaults)
    print("\n1. Minimal quick inject (auto-creates group):")
    minimal_request = {
        "group_name": "quick_test",
        "title": "First Document",
        "content": "This is a quick test of the auto-inject feature. It should create the group automatically."
    }
    response = requests.post(f"{BASE_URL}/quick-inject", json=minimal_request)
    print_response(response)
    
    # Example 2: Full request with all options
    print("\n2. Full quick inject with all options:")
    full_request = {
        "group_name": "physics_quick",
        "scope": "global",
        "owner": None,
        "group_description": "Physics papers via quick inject",
        "embed_model": "nomic-embed-text",
        "title": "Quantum Computing Review",
        "content": """
        This comprehensive review discusses the current state of quantum computing.
        We cover qubit implementations, error correction, and quantum algorithms.
        Recent advances in superconducting qubits show promising results.
        """,
        "metadata": {
            "author": "Dr. Jane Doe",
            "institution": "Quantum Research Lab",
            "year": 2024,
            "tags": ["quantum", "computing", "review"]
        },
        "chunk_size": 500,
        "chunk_overlap": 100
    }
    response = requests.post(f"{BASE_URL}/quick-inject", json=full_request)
    print_response(response)
    
    # Example 3: Inject to same group (tests reuse of existing group)
    print("\n3. Inject another document to same group:")
    second_doc = {
        "group_name": "physics_quick",
        "title": "Quantum Error Correction",
        "content": "Error correction is essential for practical quantum computing. This paper reviews various approaches.",
        "metadata": {"topic": "error_correction"}
    }
    response = requests.post(f"{BASE_URL}/quick-inject", json=second_doc)
    print_response(response)
    
    # Example 4: Test duplicate detection
    print("\n4. Try to inject duplicate (should be skipped):")
    response = requests.post(f"{BASE_URL}/quick-inject", json=second_doc)
    print_response(response)


# =========================
# Example 3: Create RAG Groups Manually
# =========================
def example_create_groups():
    """Create multiple RAG groups with different embedding models."""
    print("\n" + "=" * 60)
    print("Example 2: Create RAG Groups")
    print("=" * 60)
    
    groups = [
        {
            "name": "physics_papers",
            "scope": "global",
            "description": "Physics research papers",
            "embed_model": "nomic-embed-text"
        },
        {
            "name": "cs_papers",
            "scope": "global",
            "description": "Computer science papers",
            "embed_model": "nomic-embed-text"
        },
        {
            "name": "user_docs",
            "scope": "user",
            "owner": "alice@example.com",
            "description": "Alice's personal documents",
            "embed_model": "nomic-embed-text"
        }
    ]
    
    for group in groups:
        print(f"\nCreating group: {group['name']}")
        response = requests.post(f"{BASE_URL}/groups", json=group)
        print_response(response)


# =========================
# Example 4: List RAG Groups
# =========================
def example_list_groups():
    """List all RAG groups."""
    print("\n" + "=" * 60)
    print("Example 3: List RAG Groups")
    print("=" * 60)
    
    # List all global groups
    print("\nGlobal groups:")
    response = requests.get(f"{BASE_URL}/groups?scope=global")
    print_response(response)
    
    # List user groups for specific owner
    print("\nUser groups for alice@example.com:")
    response = requests.get(
        f"{BASE_URL}/groups",
        params={"scope": "user", "owner": "alice@example.com"}
    )
    print_response(response)


# =========================
# Example 5: Get Specific Group
# =========================
def example_get_group():
    """Get details of a specific RAG group."""
    print("\n" + "=" * 60)
    print("Example 5: Get Specific Group")
    print("=" * 60)
    
    response = requests.get(
        f"{BASE_URL}/groups/physics_papers",
        params={"scope": "global"}
    )
    print_response(response)


# =========================
# Example 6: Inject Single Document
# =========================
def example_inject_single_document():
    """Inject a single document into a RAG group."""
    print("\n" + "=" * 60)
    print("Example 6: Inject Single Document")
    print("=" * 60)
    
    document = {
        "title": "Quantum Entanglement in Multi-Particle Systems",
        "content": """
        Abstract: This paper presents a comprehensive study of quantum entanglement
        in multi-particle systems. We explore the mathematical framework and
        experimental verification of entangled states.
        
        Introduction: Quantum entanglement is a fundamental phenomenon in quantum
        mechanics where particles become correlated in such a way that the quantum
        state of each particle cannot be described independently.
        
        Methods: We used advanced quantum optical techniques to generate and
        measure entangled photon pairs. The experimental setup consisted of a
        spontaneous parametric down-conversion source.
        
        Results: Our measurements demonstrate strong correlations between
        entangled particles, violating Bell's inequalities by 15 standard
        deviations.
        
        Conclusion: These results provide further evidence for the non-local
        nature of quantum mechanics and have implications for quantum computing
        and cryptography.
        """,
        "metadata": {
            "author": "Dr. Jane Smith",
            "institution": "MIT",
            "year": 2024,
            "doi": "10.1234/quantum.2024.001",
            "tags": ["quantum mechanics", "entanglement", "experimental physics"],
            "citations": 15
        },
        "chunk_size": 500,
        "chunk_overlap": 100
    }
    
    response = requests.post(
        f"{BASE_URL}/inject/physics_papers",
        params={"scope": "global"},
        json=document
    )
    print_response(response)


# =========================
# Example 7: Batch Inject Documents
# =========================
def example_batch_inject():
    """Inject multiple documents in a single request."""
    print("\n" + "=" * 60)
    print("Example 7: Batch Inject Documents")
    print("=" * 60)
    
    documents = [
        {
            "title": "Machine Learning for Physics",
            "content": "This paper explores the application of machine learning techniques to physics problems. We demonstrate how neural networks can predict particle trajectories and solve complex differential equations.",
            "metadata": {
                "author": "Dr. Bob Johnson",
                "year": 2024,
                "tags": ["machine learning", "physics", "neural networks"]
            }
        },
        {
            "title": "Topological Quantum Computing",
            "content": "Topological quantum computing uses anyons and braiding operations to perform quantum computations. This approach provides inherent protection against decoherence.",
            "metadata": {
                "author": "Dr. Alice Chen",
                "year": 2023,
                "tags": ["quantum computing", "topology", "anyons"]
            }
        },
        {
            "title": "Dark Matter Detection Methods",
            "content": "We review current experimental approaches to dark matter detection, including direct detection experiments, indirect detection via astrophysical observations, and collider searches.",
            "metadata": {
                "author": "Dr. Carlos Rodriguez",
                "year": 2024,
                "tags": ["dark matter", "cosmology", "experimental physics"]
            }
        }
    ]
    
    batch = {
        "documents": documents,
        "chunk_size": 800,
        "chunk_overlap": 150
    }
    
    response = requests.post(
        f"{BASE_URL}/inject/physics_papers/batch",
        params={"scope": "global"},
        json=batch
    )
    print_response(response)


# =========================
# Example 8: List Documents in Group
# =========================
def example_list_documents():
    """List all documents in a RAG group."""
    print("\n" + "=" * 60)
    print("Example 8: List Documents in Group")
    print("=" * 60)
    
    response = requests.get(
        f"{BASE_URL}/documents/physics_papers",
        params={
            "scope": "global",
            "limit": 10,
            "offset": 0
        }
    )
    print_response(response)


# =========================
# Example 9: Inject from URL Crawler
# =========================
def example_inject_from_url_crawler():
    """Example of injecting documents from URL crawler results."""
    print("\n" + "=" * 60)
    print("Example 9: Inject from URL Crawler")
    print("=" * 60)
    
    # Simulate fetching crawled URLs (would come from backend web API)
    crawled_urls = [
        {
            "id": "url-123",
            "url": "https://arxiv.org/abs/2401.00001",
            "title": "Novel Approach to String Theory",
            "content": "Full text content from arXiv paper...",
            "content_type": "application/pdf",
            "created_at": "2024-01-15T10:00:00"
        },
        {
            "id": "url-124",
            "url": "https://inspirehep.net/literature/12345",
            "title": "Higgs Boson Properties",
            "content": "Full text content from INSPIRE paper...",
            "content_type": "text/html",
            "created_at": "2024-01-15T11:00:00"
        }
    ]
    
    # Prepare documents for batch injection
    documents = []
    for url_data in crawled_urls:
        documents.append({
            "title": url_data["title"],
            "content": url_data["content"],
            "url_id": url_data["id"],
            "metadata": {
                "url": url_data["url"],
                "crawled_at": url_data["created_at"],
                "content_type": url_data.get("content_type"),
                "source": "url_crawler"
            }
        })
    
    # Batch inject
    response = requests.post(
        f"{BASE_URL}/inject/physics_papers/batch",
        json={"documents": documents}
    )
    print_response(response)


# =========================
# Example 10: Custom Chunking Strategies
# =========================
def example_custom_chunking():
    """Demonstrate different chunking strategies for different content types."""
    print("\n" + "=" * 60)
    print("Example 10: Custom Chunking Strategies")
    print("=" * 60)
    
    # Strategy 1: Small chunks for dense technical content
    print("\nStrategy 1: Small chunks for dense content")
    dense_doc = {
        "title": "Quantum Field Theory Equations",
        "content": "The Lagrangian density for QED is L = -1/4 F_μν F^μν + ψ̄(iγ^μ D_μ - m)ψ...",
        "chunk_size": 300,
        "chunk_overlap": 50,
        "metadata": {"type": "equations"}
    }
    response = requests.post(f"{BASE_URL}/inject/physics_papers", json=dense_doc)
    print_response(response)
    
    # Strategy 2: Large chunks for narrative content
    print("\nStrategy 2: Large chunks for narrative content")
    narrative_doc = {
        "title": "History of Quantum Mechanics",
        "content": "The development of quantum mechanics in the early 20th century...",
        "chunk_size": 2000,
        "chunk_overlap": 400,
        "metadata": {"type": "narrative"}
    }
    response = requests.post(f"{BASE_URL}/inject/physics_papers", json=narrative_doc)
    print_response(response)


# =========================
# Example 11: Multi-tenant Document Management
# =========================
def example_multi_tenant():
    """Demonstrate multi-tenant document management."""
    print("\n" + "=" * 60)
    print("Example 11: Multi-tenant Document Management")
    print("=" * 60)
    
    # User 1: Create personal group and inject document
    print("\nUser 1 (alice@example.com):")
    alice_group = {
        "name": "alice_research",
        "scope": "user",
        "owner": "alice@example.com",
        "description": "Alice's research notes",
        "embed_model": "nomic-embed-text"
    }
    response = requests.post(f"{BASE_URL}/groups", json=alice_group)
    print_response(response)
    
    alice_doc = {
        "title": "My Research Notes",
        "content": "Personal notes on quantum computing research...",
        "metadata": {"private": True}
    }
    response = requests.post(
        f"{BASE_URL}/inject/alice_research",
        params={"scope": "user"},
        json=alice_doc
    )
    print_response(response)
    
    # User 2: Create separate group
    print("\nUser 2 (bob@example.com):")
    bob_group = {
        "name": "bob_notes",
        "scope": "user",
        "owner": "bob@example.com",
        "description": "Bob's notes",
        "embed_model": "nomic-embed-text"
    }
    response = requests.post(f"{BASE_URL}/groups", json=bob_group)
    print_response(response)


# =========================
# Example 12: Update Group Metadata
# =========================
def example_update_group():
    """Update RAG group metadata."""
    print("\n" + "=" * 60)
    print("Example 12: Update Group Metadata")
    print("=" * 60)
    
    update = {
        "description": "Updated description: High-energy physics papers",
        "owner": "physics-team@example.com"
    }
    
    response = requests.patch(
        f"{BASE_URL}/groups/physics_papers",
        params={"scope": "global"},
        json=update
    )
    print_response(response)


# =========================
# Example 13: Error Handling
# =========================
def example_error_handling():
    """Demonstrate common error scenarios."""
    print("\n" + "=" * 60)
    print("Example 13: Error Handling")
    print("=" * 60)
    
    # Error 1: Group already exists
    print("\nError 1: Creating duplicate group")
    duplicate_group = {
        "name": "physics_papers",
        "scope": "global",
        "embed_model": "nomic-embed-text"
    }
    response = requests.post(f"{BASE_URL}/groups", json=duplicate_group)
    print_response(response)
    
    # Error 2: Group not found
    print("\nError 2: Getting non-existent group")
    response = requests.get(f"{BASE_URL}/groups/nonexistent")
    print_response(response)
    
    # Error 3: Injecting to non-existent group
    print("\nError 3: Injecting to non-existent group")
    doc = {
        "title": "Test",
        "content": "Test content"
    }
    response = requests.post(f"{BASE_URL}/inject/nonexistent", json=doc)
    print_response(response)


# =========================
# Example 14: Complete RAG Workflow
# =========================
def example_complete_workflow():
    """Demonstrate complete RAG workflow: inject → query."""
    print("\n" + "=" * 60)
    print("Example 14: Complete RAG Workflow")
    print("=" * 60)
    
    # Step 1: Create group
    print("\nStep 1: Create RAG group")
    group = {
        "name": "demo_workflow",
        "scope": "global",
        "description": "Demo for complete workflow",
        "embed_model": "nomic-embed-text"
    }
    response = requests.post(f"{BASE_URL}/groups", json=group)
    print_response(response)
    
    # Step 2: Inject documents
    print("\nStep 2: Inject documents")
    documents = [
        {
            "title": "Quantum Computing Basics",
            "content": "Quantum computing uses quantum bits or qubits that can exist in superposition states.",
            "metadata": {"topic": "basics"}
        },
        {
            "title": "Quantum Algorithms",
            "content": "Shor's algorithm and Grover's algorithm are fundamental quantum algorithms.",
            "metadata": {"topic": "algorithms"}
        }
    ]
    response = requests.post(
        f"{BASE_URL}/inject/demo_workflow/batch",
        json={"documents": documents}
    )
    print_response(response)
    
    # Step 3: Query via RAG MCP (port 4001)
    print("\nStep 3: Query via RAG MCP service")
    print("(Would call http://localhost:4001/rag_search)")
    print("Example query: 'quantum algorithms' → retrieves relevant chunks")


# =========================
# Example 15: Delete Operations
# =========================
def example_delete_operations():
    """Demonstrate delete operations."""
    print("\n" + "=" * 60)
    print("Example 15: Delete Operations")
    print("=" * 60)
    
    # First, create a test group and document
    print("\nCreating test group and document...")
    test_group = {
        "name": "test_delete",
        "scope": "global",
        "embed_model": "nomic-embed-text"
    }
    response = requests.post(f"{BASE_URL}/groups", json=test_group)
    
    test_doc = {
        "title": "Test Document",
        "content": "This is a test document for deletion."
    }
    response = requests.post(f"{BASE_URL}/inject/test_delete", json=test_doc)
    doc_id = response.json()["id"]
    print(f"Created document: {doc_id}")
    
    # Delete document
    print(f"\nDeleting document {doc_id}...")
    response = requests.delete(f"{BASE_URL}/documents/{doc_id}")
    print_response(response)
    
    # Delete group
    print("\nDeleting group...")
    response = requests.delete(f"{BASE_URL}/groups/test_delete")
    print_response(response)


# =========================
# Run All Examples
# =========================
def run_all_examples():
    """Run all examples in sequence."""
    examples = [
        ("Health Check", example_health_check),
        ("Quick Inject", example_quick_inject),
        ("Create Groups", example_create_groups),
        ("List Groups", example_list_groups),
        ("Get Group", example_get_group),
        ("Inject Single Document", example_inject_single_document),
        ("Batch Inject", example_batch_inject),
        ("List Documents", example_list_documents),
        ("Inject from URL Crawler", example_inject_from_url_crawler),
        ("Custom Chunking", example_custom_chunking),
        ("Multi-tenant", example_multi_tenant),
        ("Update Group", example_update_group),
        ("Error Handling", example_error_handling),
        ("Complete Workflow", example_complete_workflow),
        ("Delete Operations", example_delete_operations),
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING ALL EXAMPLES")
    print("=" * 60)
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    # Run individual examples or all at once
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        example_map = {
            "1": example_health_check,
            "2": example_quick_inject,
            "3": example_create_groups,
            "4": example_list_groups,
            "5": example_get_group,
            "6": example_inject_single_document,
            "7": example_batch_inject,
            "8": example_list_documents,
            "9": example_inject_from_url_crawler,
            "10": example_custom_chunking,
            "11": example_multi_tenant,
            "12": example_update_group,
            "13": example_error_handling,
            "14": example_complete_workflow,
            "15": example_delete_operations,
        }
        
        if example_num in example_map:
            example_map[example_num]()
        else:
            print("Usage: python examples.py [1-15]")
            print("Or run without arguments to execute all examples")
    else:
        run_all_examples()
