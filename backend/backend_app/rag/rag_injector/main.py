"""
RAG Injection REST Service

A FastAPI REST service for injecting documents into the RAG knowledge base.
Runs on port 4002 and provides endpoints for:
- Creating/managing RAG groups
- Injecting documents
- Managing embeddings
- Querying injection status
"""
import os
import hashlib
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psycopg

# Import common RAG utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rag_common import (
    DATABASE_URL, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, COLLECTION_DOCS, PG_DSN,
    LANGCHAIN_AVAILABLE, get_embeddings_for_model, get_vector_store_for_model,
    db_exec
)

# LangChain imports for text splitting and documents
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError as e:
    LANGCHAIN_ERROR = str(e)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Injection Service",
    description="REST API for injecting documents into RAG knowledge base",
    version="1.0.0"
)

def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# =========================
# Pydantic Models
# =========================
class RAGGroupCreate(BaseModel):
    name: str = Field(..., description="Unique name for the RAG group")
    scope: str = Field("global", description="Scope for multi-tenancy")
    owner: Optional[str] = Field(None, description="Owner identifier")
    description: Optional[str] = Field(None, description="Group description")
    embed_model: str = Field(..., description="Embedding model to use (e.g., nomic-embed-text)")

class RAGGroupUpdate(BaseModel):
    description: Optional[str] = None
    owner: Optional[str] = None

class RAGGroupResponse(BaseModel):
    id: str
    name: str
    scope: str
    owner: Optional[str]
    description: Optional[str]
    embed_model: str
    doc_count: int
    created_at: str
    updated_at: str

class DocumentInject(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    url_id: Optional[str] = Field(None, description="Optional source URL ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    chunk_size: int = Field(1000, description="Chunk size for text splitting")
    chunk_overlap: int = Field(200, description="Overlap between chunks")

class DocumentBatchInject(BaseModel):
    documents: List[DocumentInject] = Field(..., description="List of documents to inject")
    chunk_size: int = Field(1000, description="Default chunk size")
    chunk_overlap: int = Field(200, description="Default chunk overlap")

class DocumentResponse(BaseModel):
    id: str
    rag_group_id: str
    url_id: Optional[str]
    title: str
    content_hash: str
    metadata: Dict[str, Any]
    created_at: str
    chunks_created: int

class InjectionStatus(BaseModel):
    document_id: str
    status: str  # pending, processing, completed, failed
    chunks_created: int
    error: Optional[str]


# =========================
# RAG Group Endpoints
# =========================
@app.get("/")
async def root():
    """Service information."""
    return {
        "service": "RAG Injection Service",
        "version": "1.0.0",
        "description": "REST API for injecting documents into RAG knowledge base",
        "endpoints": {
            "groups": "/groups",
            "inject": "/inject/{rag_group_name}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "langchain_available": LANGCHAIN_AVAILABLE,
        "database_connected": True,  # TODO: Add actual DB check
        "embed_model": OLLAMA_EMBED_MODEL,
        "collection": COLLECTION_DOCS
    }

@app.get("/groups", response_model=List[RAGGroupResponse])
async def list_groups(scope: str = "global", owner: Optional[str] = None):
    """List all RAG groups with optional filtering."""
    try:
        query = """
            SELECT id, name, scope, owner, description, embed_model,
                   doc_count, created_at, updated_at
            FROM rag_groups
            WHERE scope = %s
        """
        params = [scope]
        
        if owner:
            query += " AND owner = %s"
            params.append(owner)
        
        query += " ORDER BY name"
        
        rows = db_exec(query, tuple(params))
        
        groups = []
        for row in rows:
            groups.append(RAGGroupResponse(
                id=row[0],
                name=row[1],
                scope=row[2],
                owner=row[3],
                description=row[4],
                embed_model=row[5],
                doc_count=row[6],
                created_at=str(row[7]),
                updated_at=str(row[8])
            ))
        
        return groups
        
    except Exception as e:
        logger.error(f"List groups error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/groups", response_model=RAGGroupResponse)
async def create_group(group: RAGGroupCreate):
    """Create a new RAG group."""
    try:
        # Check if name already exists in scope
        existing = db_exec(
            "SELECT id FROM rag_groups WHERE name = %s AND scope = %s",
            (group.name, group.scope)
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"RAG group '{group.name}' already exists in scope '{group.scope}'"
            )
        
        # Create new group
        group_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        db_exec("""
            INSERT INTO rag_groups (id, name, scope, owner, description, embed_model, doc_count, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s)
        """, (
            group_id,
            group.name,
            group.scope,
            group.owner,
            group.description,
            group.embed_model,
            now,
            now
        ))
        
        logger.info(f"Created RAG group: {group.name} (id: {group_id}) with model: {group.embed_model}")
        
        return RAGGroupResponse(
            id=group_id,
            name=group.name,
            scope=group.scope,
            owner=group.owner,
            description=group.description,
            embed_model=group.embed_model,
            doc_count=0,
            created_at=str(now),
            updated_at=str(now)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create group error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/groups/{group_name}", response_model=RAGGroupResponse)
async def get_group(group_name: str, scope: str = "global"):
    """Get a specific RAG group by name."""
    try:
        rows = db_exec("""
            SELECT id, name, scope, owner, description, embed_model,
                   doc_count, created_at, updated_at
            FROM rag_groups
            WHERE name = %s AND scope = %s
        """, (group_name, scope))
        
        if not rows:
            raise HTTPException(status_code=404, detail=f"RAG group '{group_name}' not found")
        
        row = rows[0]
        return RAGGroupResponse(
            id=row[0],
            name=row[1],
            scope=row[2],
            owner=row[3],
            description=row[4],
            embed_model=row[5],
            doc_count=row[6],
            created_at=str(row[7]),
            updated_at=str(row[8])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get group error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/groups/{group_name}")
async def update_group(group_name: str, update: RAGGroupUpdate, scope: str = "global"):
    """Update RAG group metadata."""
    try:
        # Check if exists
        rows = db_exec(
            "SELECT id FROM rag_groups WHERE name = %s AND scope = %s",
            (group_name, scope)
        )
        if not rows:
            raise HTTPException(status_code=404, detail=f"RAG group '{group_name}' not found")
        
        # Build update query
        updates = []
        params = []
        
        if update.description is not None:
            updates.append("description = %s")
            params.append(update.description)
        
        if update.owner is not None:
            updates.append("owner = %s")
            params.append(update.owner)
        
        if updates:
            updates.append("updated_at = %s")
            params.append(datetime.utcnow())
            params.extend([group_name, scope])
            
            query = f"UPDATE rag_groups SET {', '.join(updates)} WHERE name = %s AND scope = %s"
            db_exec(query, tuple(params))
        
        return {"status": "updated", "group": group_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update group error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/groups/{group_name}")
async def delete_group(group_name: str, scope: str = "global"):
    """Delete a RAG group and all its documents."""
    try:
        # Get group ID
        rows = db_exec(
            "SELECT id FROM rag_groups WHERE name = %s AND scope = %s",
            (group_name, scope)
        )
        if not rows:
            raise HTTPException(status_code=404, detail=f"RAG group '{group_name}' not found")
        
        group_id = rows[0][0]
        
        # Delete documents (cascades to embeddings via ON DELETE CASCADE)
        db_exec("DELETE FROM rag_documents WHERE rag_group_id = %s", (group_id,))
        
        # Delete group
        db_exec("DELETE FROM rag_groups WHERE id = %s", (group_id,))
        
        logger.info(f"Deleted RAG group: {group_name} (id: {group_id})")
        
        return {"status": "deleted", "group": group_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete group error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Document Injection Endpoints
# =========================
def inject_document_to_vectorstore(
    rag_group_id: str,
    rag_group_name: str,
    embed_model: str,
    doc_id: str,
    title: str,
    content: str,
    metadata: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int
) -> int:
    """Inject document into vector store with chunking."""
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("LangChain not available")
    
    # Get vector store for this embedding model
    vector_store = get_vector_store_for_model(embed_model)
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Split content into chunks
    chunks = text_splitter.split_text(content)
    logger.info(f"Split document '{title}' into {len(chunks)} chunks")
    
    # Create Document objects with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        doc_metadata = {
            **metadata,
            "rag_group": rag_group_name,
            "rag_group_id": rag_group_id,
            "document_id": doc_id,
            "title": title,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        documents.append(Document(page_content=chunk, metadata=doc_metadata))
    
    # Add to vector store
    vector_store.add_documents(documents)
    logger.info(f"Added {len(documents)} document chunks to vector store")
    
    return len(chunks)

@app.post("/inject/{rag_group_name}", response_model=DocumentResponse)
async def inject_document(
    rag_group_name: str,
    document: DocumentInject,
    background_tasks: BackgroundTasks,
    scope: str = "global"
):
    """
    Inject a single document into a RAG group.
    
    The document is:
    1. Stored in the rag_documents table
    2. Chunked using RecursiveCharacterTextSplitter
    3. Embedded using the group's embedding model
    4. Stored in PGVector for similarity search
    """
    try:
        # Get RAG group
        rows = db_exec("""
            SELECT id, embed_model FROM rag_groups
            WHERE name = %s AND scope = %s
        """, (rag_group_name, scope))
        
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"RAG group '{rag_group_name}' not found in scope '{scope}'"
            )
        
        rag_group_id = rows[0][0]
        embed_model = rows[0][1]
        
        # Compute content hash
        content_hash = compute_content_hash(document.content)
        
        # Check for duplicate
        existing = db_exec("""
            SELECT id FROM rag_documents
            WHERE rag_group_id = %s AND content_hash = %s
        """, (rag_group_id, content_hash))
        
        if existing:
            raise HTTPException(
                status_code=409,
                detail="Document with same content already exists in this group"
            )
        
        # Create document record
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        db_exec("""
            INSERT INTO rag_documents (id, rag_group_id, url_id, title, content, content_hash, metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            doc_id,
            rag_group_id,
            document.url_id,
            document.title,
            document.content,
            content_hash,
            psycopg.types.json.Json(document.metadata),
            now,
            now
        ))
        
        logger.info(f"Created document record: {doc_id} in group: {rag_group_name}")
        
        # Inject into vector store (synchronously for now)
        chunks_created = inject_document_to_vectorstore(
            rag_group_id=rag_group_id,
            rag_group_name=rag_group_name,
            embed_model=embed_model,
            doc_id=doc_id,
            title=document.title,
            content=document.content,
            metadata=document.metadata,
            chunk_size=document.chunk_size,
            chunk_overlap=document.chunk_overlap
        )
        
        # Update doc count
        db_exec("""
            UPDATE rag_groups
            SET doc_count = doc_count + 1, updated_at = %s
            WHERE id = %s
        """, (datetime.utcnow(), rag_group_id))
        
        return DocumentResponse(
            id=doc_id,
            rag_group_id=rag_group_id,
            url_id=document.url_id,
            title=document.title,
            content_hash=content_hash,
            metadata=document.metadata,
            created_at=str(now),
            chunks_created=chunks_created
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inject document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inject/{rag_group_name}/batch")
async def inject_documents_batch(
    rag_group_name: str,
    batch: DocumentBatchInject,
    scope: str = "global"
):
    """
    Inject multiple documents into a RAG group.
    
    Returns status for each document.
    """
    try:
        # Get RAG group
        rows = db_exec("""
            SELECT id, embed_model FROM rag_groups
            WHERE name = %s AND scope = %s
        """, (rag_group_name, scope))
        
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"RAG group '{rag_group_name}' not found in scope '{scope}'"
            )
        
        rag_group_id = rows[0][0]
        embed_model = rows[0][1]
        
        results = []
        successful = 0
        failed = 0
        
        for doc in batch.documents:
            try:
                # Use batch defaults if not specified
                chunk_size = doc.chunk_size or batch.chunk_size
                chunk_overlap = doc.chunk_overlap or batch.chunk_overlap
                
                # Compute content hash
                content_hash = compute_content_hash(doc.content)
                
                # Check for duplicate
                existing = db_exec("""
                    SELECT id FROM rag_documents
                    WHERE rag_group_id = %s AND content_hash = %s
                """, (rag_group_id, content_hash))
                
                if existing:
                    results.append({
                        "title": doc.title,
                        "status": "skipped",
                        "reason": "duplicate content"
                    })
                    continue
                
                # Create document
                doc_id = str(uuid.uuid4())
                now = datetime.utcnow()
                
                db_exec("""
                    INSERT INTO rag_documents (id, rag_group_id, url_id, title, content, content_hash, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    doc_id,
                    rag_group_id,
                    doc.url_id,
                    doc.title,
                    doc.content,
                    content_hash,
                    psycopg.types.json.Json(doc.metadata),
                    now,
                    now
                ))
                
                # Inject into vector store
                chunks_created = inject_document_to_vectorstore(
                    rag_group_id=rag_group_id,
                    rag_group_name=rag_group_name,
                    embed_model=embed_model,
                    doc_id=doc_id,
                    title=doc.title,
                    content=doc.content,
                    metadata=doc.metadata,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                results.append({
                    "document_id": doc_id,
                    "title": doc.title,
                    "status": "completed",
                    "chunks_created": chunks_created
                })
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to inject document '{doc.title}': {e}")
                results.append({
                    "title": doc.title,
                    "status": "failed",
                    "error": str(e)
                })
                failed += 1
        
        # Update doc count
        if successful > 0:
            db_exec("""
                UPDATE rag_groups
                SET doc_count = doc_count + %s, updated_at = %s
                WHERE id = %s
            """, (successful, datetime.utcnow(), rag_group_id))
        
        return {
            "rag_group": rag_group_name,
            "total": len(batch.documents),
            "successful": successful,
            "failed": failed,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch inject error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{rag_group_name}")
async def list_documents(
    rag_group_name: str,
    scope: str = "global",
    limit: int = 20,
    offset: int = 0
):
    """List documents in a RAG group."""
    try:
        # Get RAG group ID
        rows = db_exec("""
            SELECT id FROM rag_groups
            WHERE name = %s AND scope = %s
        """, (rag_group_name, scope))
        
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"RAG group '{rag_group_name}' not found"
            )
        
        rag_group_id = rows[0][0]
        
        # Get documents
        docs = db_exec("""
            SELECT id, title, content_hash, metadata, created_at,
                   LENGTH(content) as content_length
            FROM rag_documents
            WHERE rag_group_id = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (rag_group_id, limit, offset))
        
        # Get total count
        count_result = db_exec("""
            SELECT COUNT(*) FROM rag_documents
            WHERE rag_group_id = %s
        """, (rag_group_id,))
        total = count_result[0][0] if count_result else 0
        
        documents = []
        for doc in docs:
            documents.append({
                "id": doc[0],
                "title": doc[1],
                "content_hash": doc[2],
                "metadata": doc[3],
                "created_at": str(doc[4]),
                "content_length": doc[5]
            })
        
        return {
            "rag_group": rag_group_name,
            "total": total,
            "limit": limit,
            "offset": offset,
            "documents": documents
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document and its embeddings."""
    try:
        # Get document info
        rows = db_exec("""
            SELECT d.rag_group_id, g.name
            FROM rag_documents d
            JOIN rag_groups g ON d.rag_group_id = g.id
            WHERE d.id = %s
        """, (document_id,))
        
        if not rows:
            raise HTTPException(status_code=404, detail="Document not found")
        
        rag_group_id = rows[0][0]
        rag_group_name = rows[0][1]
        
        # Delete document (embeddings cascade via metadata filter)
        # Note: PGVector doesn't auto-delete by foreign key, need manual cleanup
        db_exec("DELETE FROM rag_documents WHERE id = %s", (document_id,))
        
        # TODO: Delete embeddings from vector store
        # vector_store.delete(filter={"document_id": document_id})
        
        # Update doc count
        db_exec("""
            UPDATE rag_groups
            SET doc_count = doc_count - 1, updated_at = %s
            WHERE id = %s
        """, (datetime.utcnow(), rag_group_id))
        
        logger.info(f"Deleted document: {document_id} from group: {rag_group_name}")
        
        return {
            "status": "deleted",
            "document_id": document_id,
            "rag_group": rag_group_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Convenience Endpoint: Auto-create Group & Inject
# =========================
class QuickInjectRequest(BaseModel):
    """Request for quick inject - automatically creates group if needed."""
    # Group info
    group_name: str = Field(..., description="RAG group name")
    scope: Optional[str] = Field(None, description="Scope (defaults to 'global' if null)")
    owner: Optional[str] = Field(None, description="Owner identifier")
    group_description: Optional[str] = Field(None, description="Group description")
    embed_model: str = Field("nomic-embed-text", description="Embedding model (default: nomic-embed-text)")
    
    # Document info
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    url_id: Optional[str] = Field(None, description="Optional source URL ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    chunk_size: int = Field(1000, description="Chunk size for text splitting")
    chunk_overlap: int = Field(200, description="Overlap between chunks")

@app.post("/quick-inject")
async def quick_inject(request: QuickInjectRequest):
    """
    Convenience endpoint: Create RAG group (if not exists) and inject document in one call.
    
    - If scope is null, defaults to "global"
    - If RAG group doesn't exist, creates it automatically
    - Then injects the document with automatic chunking
    
    This is ideal for simple workflows where you want to quickly add documents
    without managing groups separately.
    """
    try:
        # Default scope to "global" if not provided
        scope = request.scope if request.scope else "global"
        
        # Check if RAG group exists
        existing_group = db_exec("""
            SELECT id, embed_model FROM rag_groups
            WHERE name = %s AND scope = %s
        """, (request.group_name, scope))
        
        if not existing_group:
            # Create new group
            group_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            db_exec("""
                INSERT INTO rag_groups (id, name, scope, owner, description, embed_model, doc_count, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s)
            """, (
                group_id,
                request.group_name,
                scope,
                request.owner,
                request.group_description,
                request.embed_model,
                now,
                now
            ))
            
            logger.info(f"Auto-created RAG group: {request.group_name} (id: {group_id}) with model: {request.embed_model}")
            embed_model = request.embed_model
            group_created = True
        else:
            group_id = existing_group[0][0]
            embed_model = existing_group[0][1]
            group_created = False
            logger.info(f"Using existing RAG group: {request.group_name} (id: {group_id})")
        
        # Compute content hash for deduplication
        content_hash = compute_content_hash(request.content)
        
        # Check for duplicate document
        existing_doc = db_exec("""
            SELECT id FROM rag_documents
            WHERE rag_group_id = %s AND content_hash = %s
        """, (group_id, content_hash))
        
        if existing_doc:
            return {
                "status": "skipped",
                "reason": "Document with same content already exists in this group",
                "group_created": group_created,
                "rag_group": request.group_name,
                "rag_group_id": group_id,
                "existing_document_id": existing_doc[0][0]
            }
        
        # Create document record
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        db_exec("""
            INSERT INTO rag_documents (id, rag_group_id, url_id, title, content, content_hash, metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            doc_id,
            group_id,
            request.url_id,
            request.title,
            request.content,
            content_hash,
            psycopg.types.json.Json(request.metadata),
            now,
            now
        ))
        
        logger.info(f"Created document record: {doc_id} in group: {request.group_name}")
        
        # Inject into vector store
        chunks_created = inject_document_to_vectorstore(
            rag_group_id=group_id,
            rag_group_name=request.group_name,
            embed_model=embed_model,
            doc_id=doc_id,
            title=request.title,
            content=request.content,
            metadata=request.metadata,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        # Update doc count
        db_exec("""
            UPDATE rag_groups
            SET doc_count = doc_count + 1, updated_at = %s
            WHERE id = %s
        """, (datetime.utcnow(), group_id))
        
        return {
            "status": "success",
            "group_created": group_created,
            "rag_group": request.group_name,
            "rag_group_id": group_id,
            "scope": scope,
            "embed_model": embed_model,
            "document": {
                "id": doc_id,
                "title": request.title,
                "content_hash": content_hash,
                "chunks_created": chunks_created,
                "created_at": str(now)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick inject error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAG Injection Service on http://0.0.0.0:4002")
    uvicorn.run(app, host="0.0.0.0", port=4002)
