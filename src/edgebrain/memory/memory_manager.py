"""
Memory Management System

This module is responsible for storing and retrieving information relevant to agent operations.
This includes short-term context for ongoing tasks and long-term knowledge for persistent learning.
"""

import asyncio
import json
import sqlite3
import uuid
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Memory(BaseModel):
    """Represents a memory entry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="ID of the agent that created this memory")
    content: str = Field(..., description="Content of the memory")
    memory_type: str = Field(..., description="Type of memory (e.g., 'goal', 'action', 'observation')")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    importance: float = Field(default=1.0, description="Importance score (0.0 to 1.0)")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the content")


class MemoryQuery(BaseModel):
    """Represents a query for retrieving memories."""
    agent_id: Optional[str] = None
    memory_type: Optional[str] = None
    content_query: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    importance_threshold: float = 0.0
    limit: int = 10


class MemoryManager:
    """
    Memory Management System for agents.
    
    Handles both short-term context and long-term knowledge storage,
    with support for semantic search through vector embeddings.
    """
    
    def __init__(self, db_path: str = "agent_memory.db", embedding_dim: int = 384):
        """
        Initialize the memory manager.
        
        Args:
            db_path: Path to the SQLite database file
            embedding_dim: Dimension of the embedding vectors
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._connection: Optional[sqlite3.Connection] = None
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Memory manager initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                importance REAL DEFAULT 1.0,
                embedding BLOB
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON memories(agent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
        
        conn.commit()
        conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        if not self._connection:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
            # Ensure tables exist for in-memory databases
            if self.db_path == ":memory:":
                self._init_database_tables(self._connection)
        return self._connection
    
    def _init_database_tables(self, conn: sqlite3.Connection):
        """Initialize database tables on the given connection."""
        cursor = conn.cursor()
        
        # Create memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                importance REAL DEFAULT 1.0,
                embedding BLOB
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON memories(agent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
        
        conn.commit()
    
    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding to bytes for storage."""
        return np.array(embedding, dtype=np.float32).tobytes()
    
    def _deserialize_embedding(self, embedding_bytes: bytes) -> List[float]:
        """Deserialize embedding from bytes."""
        return np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        This is a mock implementation. In a real system, you would use
        a proper embedding model like sentence-transformers.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        # Mock embedding generation (replace with actual embedding model)
        # For now, we'll create a simple hash-based embedding
        import hashlib
        
        # Create a deterministic "embedding" based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numbers and normalize
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(value)
        
        # Pad or truncate to desired dimension
        while len(embedding) < self.embedding_dim:
            embedding.extend(embedding[:self.embedding_dim - len(embedding)])
        
        return embedding[:self.embedding_dim]
    
    async def store_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0
    ) -> str:
        """
        Store a new memory.
        
        Args:
            agent_id: ID of the agent creating the memory
            content: Content of the memory
            memory_type: Type of memory
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            
        Returns:
            ID of the created memory
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})
        
        # Generate embedding
        embedding = await self._generate_embedding(content)
        embedding_bytes = self._serialize_embedding(embedding)
        
        # Store in database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO memories (id, agent_id, content, memory_type, timestamp, metadata, importance, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, agent_id, content, memory_type, timestamp, metadata_json, importance, embedding_bytes))
        
        conn.commit()
        
        logger.debug(f"Stored memory {memory_id} for agent {agent_id}")
        return memory_id
    
    async def retrieve_memories(
        self,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        query: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        importance_threshold: float = 0.0,
        limit: int = 10
    ) -> List[str]:
        """
        Retrieve memories based on criteria.
        
        Args:
            agent_id: Filter by agent ID
            memory_type: Filter by memory type
            query: Semantic query for content search
            time_range: Filter by time range
            importance_threshold: Minimum importance score
            limit: Maximum number of memories to return
            
        Returns:
            List of memory contents
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build query
        sql_parts = ["SELECT content, importance, embedding FROM memories WHERE 1=1"]
        params = []
        
        if agent_id:
            sql_parts.append("AND agent_id = ?")
            params.append(agent_id)
        
        if memory_type:
            sql_parts.append("AND memory_type = ?")
            params.append(memory_type)
        
        if time_range:
            sql_parts.append("AND timestamp BETWEEN ? AND ?")
            params.extend([time_range[0].isoformat(), time_range[1].isoformat()])
        
        if importance_threshold > 0:
            sql_parts.append("AND importance >= ?")
            params.append(importance_threshold)
        
        sql_parts.append("ORDER BY importance DESC, timestamp DESC")
        sql_parts.append("LIMIT ?")
        params.append(limit * 2)  # Get more for semantic filtering
        
        sql = " ".join(sql_parts)
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        memories = []
        for row in rows:
            content = row["content"]
            importance = row["importance"]
            embedding_bytes = row["embedding"]
            
            memory_data = {
                "content": content,
                "importance": importance,
                "embedding": self._deserialize_embedding(embedding_bytes) if embedding_bytes else None
            }
            memories.append(memory_data)
        
        # If we have a semantic query, rank by similarity
        if query and memories:
            query_embedding = await self._generate_embedding(query)
            
            # Calculate similarities
            for memory in memories:
                if memory["embedding"]:
                    similarity = self._cosine_similarity(query_embedding, memory["embedding"])
                    memory["similarity"] = similarity
                else:
                    memory["similarity"] = 0.0
            
            # Sort by similarity and importance
            memories.sort(key=lambda x: (x["similarity"], x["importance"]), reverse=True)
        
        # Return just the content, limited to requested amount
        return [memory["content"] for memory in memories[:limit]]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory object or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        embedding = self._deserialize_embedding(row["embedding"]) if row["embedding"] else None
        
        return Memory(
            id=row["id"],
            agent_id=row["agent_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=metadata,
            importance=row["importance"],
            embedding=embedding
        )
    
    async def update_memory_importance(self, memory_id: str, importance: float) -> bool:
        """
        Update the importance score of a memory.
        
        Args:
            memory_id: ID of the memory to update
            importance: New importance score
            
        Returns:
            True if updated successfully, False if memory not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("UPDATE memories SET importance = ? WHERE id = ?", (importance, memory_id))
        
        if cursor.rowcount > 0:
            conn.commit()
            logger.debug(f"Updated importance for memory {memory_id} to {importance}")
            return True
        
        return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted successfully, False if memory not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        
        if cursor.rowcount > 0:
            conn.commit()
            logger.debug(f"Deleted memory {memory_id}")
            return True
        
        return False
    
    async def cleanup_old_memories(self, agent_id: str, days_old: int = 30, keep_important: bool = True) -> int:
        """
        Clean up old memories for an agent.
        
        Args:
            agent_id: ID of the agent
            days_old: Delete memories older than this many days
            keep_important: Whether to keep memories with high importance (>0.8)
            
        Returns:
            Number of memories deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        sql = "DELETE FROM memories WHERE agent_id = ? AND timestamp < ?"
        params = [agent_id, cutoff_date.isoformat()]
        
        if keep_important:
            sql += " AND importance <= 0.8"
        
        cursor.execute(sql, params)
        deleted_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"Cleaned up {deleted_count} old memories for agent {agent_id}")
        return deleted_count
    
    async def get_memory_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about stored memories.
        
        Args:
            agent_id: Optional agent ID to filter by
            
        Returns:
            Dictionary with memory statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Base query
        where_clause = ""
        params = []
        
        if agent_id:
            where_clause = "WHERE agent_id = ?"
            params.append(agent_id)
        
        # Total count
        cursor.execute(f"SELECT COUNT(*) as count FROM memories {where_clause}", params)
        total_count = cursor.fetchone()["count"]
        
        # Count by type
        cursor.execute(f"SELECT memory_type, COUNT(*) as count FROM memories {where_clause} GROUP BY memory_type", params)
        type_counts = {row["memory_type"]: row["count"] for row in cursor.fetchall()}
        
        # Average importance
        cursor.execute(f"SELECT AVG(importance) as avg_importance FROM memories {where_clause}", params)
        avg_importance = cursor.fetchone()["avg_importance"] or 0.0
        
        return {
            "total_memories": total_count,
            "memories_by_type": type_counts,
            "average_importance": avg_importance
        }
    
    def close(self):
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()

