"""
RAG MCP Server Usage Examples

This file demonstrates how to use the RAG MCP server's tools.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def example_vector_search():
    """Example: Semantic search using vector similarity."""
    print("\n=== Example 1: Vector Similarity Search ===")
    
    async with stdio_client(
        StdioServerParameters(
            command="python",
            args=["main.py"],
            env={
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/dbname",
                "OLLAMA_BASE_URL": "http://localhost:11434"
            }
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Search for machine learning content
            result = await session.call_tool("rag_search", {
                "query": "What is machine learning and how does it work?",
                "k": 5,
                "score_threshold": 0.5
            })
            
            print(json.dumps(json.loads(result.content[0].text), indent=2))


async def example_database_query():
    """Example: Database query with filters."""
    print("\n=== Example 2: Database Query ===")
    
    async with stdio_client(
        StdioServerParameters(
            command="python",
            args=["main.py"],
            env={
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/dbname"
            }
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Query documents with title pattern
            result = await session.call_tool("rag_query", {
                "title_pattern": "%python%",
                "limit": 10
            })
            
            print(json.dumps(json.loads(result.content[0].text), indent=2))


async def example_get_document():
    """Example: Get specific document by ID."""
    print("\n=== Example 3: Get Document by ID ===")
    
    async with stdio_client(
        StdioServerParameters(
            command="python",
            args=["main.py"],
            env={
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/dbname"
            }
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get document by ID
            result = await session.call_tool("rag_get_document", {
                "document_id": "d1"
            })
            
            print(json.dumps(json.loads(result.content[0].text), indent=2))


async def example_list_groups():
    """Example: List all RAG groups."""
    print("\n=== Example 4: List RAG Groups ===")
    
    async with stdio_client(
        StdioServerParameters(
            command="python",
            args=["main.py"],
            env={
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/dbname"
            }
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List all groups in global scope
            result = await session.call_tool("rag_list_groups", {
                "scope": "global"
            })
            
            print(json.dumps(json.loads(result.content[0].text), indent=2))


async def example_group_documents():
    """Example: Get all documents in a group."""
    print("\n=== Example 5: Get Group Documents ===")
    
    async with stdio_client(
        StdioServerParameters(
            command="python",
            args=["main.py"],
            env={
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/dbname"
            }
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get documents from ML documentation group
            result = await session.call_tool("rag_get_group_documents", {
                "rag_group_name": "ML Documentation",
                "limit": 20
            })
            
            print(json.dumps(json.loads(result.content[0].text), indent=2))


async def example_group_filtered_search():
    """Example: Vector search within specific group."""
    print("\n=== Example 6: Group-Filtered Vector Search ===")
    
    async with stdio_client(
        StdioServerParameters(
            command="python",
            args=["main.py"],
            env={
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/dbname",
                "OLLAMA_BASE_URL": "http://localhost:11434"
            }
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Search only within API Reference group
            result = await session.call_tool("rag_search", {
                "query": "authentication methods",
                "k": 3,
                "rag_group": "API Reference"
            })
            
            print(json.dumps(json.loads(result.content[0].text), indent=2))


async def example_resources():
    """Example: Access MCP resources."""
    print("\n=== Example 7: Access Resources ===")
    
    async with stdio_client(
        StdioServerParameters(
            command="python",
            args=["main.py"],
            env={
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/dbname"
            }
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get metadata resource
            resources = await session.list_resources()
            print("Available resources:", [r.name for r in resources.resources])
            
            # Read metadata
            metadata = await session.read_resource("rag://metadata")
            print("\nMetadata:")
            print(json.dumps(json.loads(metadata.contents[0].text), indent=2))
            
            # Read groups list
            groups = await session.read_resource("rag://groups")
            print("\nGroups:")
            print(json.dumps(json.loads(groups.contents[0].text), indent=2))


# HTTP/SSE Examples (without MCP client library)

async def example_http_call():
    """Example: Direct HTTP call to MCP server."""
    print("\n=== Example 8: HTTP/SSE Call ===")
    
    import httpx
    
    async with httpx.AsyncClient() as client:
        # Call rag_search tool
        response = await client.post(
            "http://localhost:4001/sse",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "rag_search",
                    "arguments": {
                        "query": "deep learning neural networks",
                        "k": 5
                    }
                },
                "id": 1
            },
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        print(json.dumps(result, indent=2))


async def example_discovery():
    """Example: Get server capabilities via discovery endpoint."""
    print("\n=== Example 9: Discovery Endpoint ===")
    
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:4001/.well-known/mcp")
        capabilities = response.json()
        
        print("Server Name:", capabilities["name"])
        print("Version:", capabilities["version"])
        print("\nAvailable Tools:")
        for tool in capabilities["capabilities"]["tools"]:
            print(f"  - {tool['name']}: {tool['description']}")
        
        print("\nAvailable Resources:")
        for resource in capabilities["capabilities"]["resources"]:
            print(f"  - {resource}")


# Practical Use Cases

async def use_case_qa_system():
    """Use case: Question-answering system."""
    print("\n=== Use Case: Q&A System ===")
    
    async with stdio_client(
        StdioServerParameters(
            command="python",
            args=["main.py"],
            env={
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/dbname",
                "OLLAMA_BASE_URL": "http://localhost:11434"
            }
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            questions = [
                "What is the difference between supervised and unsupervised learning?",
                "How do I configure authentication in the API?",
                "What are the best practices for model deployment?"
            ]
            
            for question in questions:
                print(f"\nQ: {question}")
                result = await session.call_tool("rag_search", {
                    "query": question,
                    "k": 3,
                    "score_threshold": 0.6
                })
                
                data = json.loads(result.content[0].text)
                if data["num_results"] > 0:
                    print(f"A: Found {data['num_results']} relevant documents")
                    for i, doc in enumerate(data["results"], 1):
                        print(f"\n  {i}. {doc['metadata'].get('title', 'Untitled')}")
                        print(f"     Score: {doc['similarity_score']:.3f}")
                        print(f"     Preview: {doc['content'][:200]}...")
                else:
                    print("A: No relevant documents found")


async def use_case_document_explorer():
    """Use case: Browse documentation by category."""
    print("\n=== Use Case: Documentation Explorer ===")
    
    async with stdio_client(
        StdioServerParameters(
            command="python",
            args=["main.py"],
            env={
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/dbname"
            }
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Step 1: List all groups
            groups_result = await session.call_tool("rag_list_groups", {
                "scope": "global"
            })
            groups_data = json.loads(groups_result.content[0].text)
            
            print("Available Documentation Categories:")
            for i, group in enumerate(groups_data["groups"], 1):
                print(f"{i}. {group['name']} ({group['doc_count']} documents)")
                if group['description']:
                    print(f"   {group['description']}")
            
            # Step 2: Explore first group
            if groups_data["groups"]:
                first_group = groups_data["groups"][0]
                print(f"\nExploring: {first_group['name']}")
                
                docs_result = await session.call_tool("rag_get_group_documents", {
                    "rag_group_name": first_group['name'],
                    "limit": 5
                })
                docs_data = json.loads(docs_result.content[0].text)
                
                print(f"Recent documents:")
                for doc in docs_data["documents"]:
                    print(f"  - {doc['title']}")
                    print(f"    ID: {doc['id']}, Created: {doc['created_at']}")


if __name__ == "__main__":
    print("RAG MCP Server Usage Examples")
    print("=" * 50)
    
    # Run examples
    asyncio.run(example_vector_search())
    asyncio.run(example_database_query())
    asyncio.run(example_get_document())
    asyncio.run(example_list_groups())
    asyncio.run(example_group_documents())
    asyncio.run(example_group_filtered_search())
    asyncio.run(example_resources())
    asyncio.run(example_http_call())
    asyncio.run(example_discovery())
    
    # Use cases
    asyncio.run(use_case_qa_system())
    asyncio.run(use_case_document_explorer())
