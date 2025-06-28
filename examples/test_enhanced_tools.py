#!/usr/bin/env python3
"""
Test script for enhanced tool registry functionality
"""

import sys
import os
import asyncio

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools.tool_registry import ToolRegistry

async def test_tools():
    """Test the enhanced tools in the registry."""
    print("🔧 Testing Enhanced Tool Registry")
    print("=" * 50)
    
    # Initialize tool registry
    registry = ToolRegistry()
    
    # Test 1: List all tools
    print("\n📋 Available Tools:")
    tools = registry.get_all_tools()
    for tool in tools:
        print(f"  - {tool.name}: {tool.description} ({tool.category})")
    
    print(f"\nTotal tools: {registry.get_tool_count()}")
    
    # Test 2: Test web search tool
    print("\n🔍 Testing Web Search Tool:")
    try:
        result = await registry.execute_tool("web_search", {
            "query": "microservices architecture",
            "max_results": 3
        })
        print(f"Search success: {result['success']}")
        print(f"Search engine: {result.get('search_engine', 'Unknown')}")
        print(f"Results count: {len(result.get('results', []))}")
        if result.get('results'):
            print(f"First result: {result['results'][0]['title']}")
    except Exception as e:
        print(f"Web search test failed: {e}")
    
    # Test 3: Test file operations
    print("\n📁 Testing File Operations:")
    try:
        # Write a test file
        write_result = await registry.execute_tool("file_write", {
            "filename": "test_output.txt",
            "content": "This is a test file created by the enhanced tool registry.\nSecond line of content."
        })
        print(f"File write success: {write_result['success']}")
        
        # Read the test file
        read_result = await registry.execute_tool("file_read", {
            "filename": "test_output.txt"
        })
        print(f"File read success: {read_result['success']}")
        print(f"File content length: {read_result.get('size', 0)} characters")
        print(f"Line count: {read_result.get('line_count', 0)}")
    except Exception as e:
        print(f"File operations test failed: {e}")
    
    # Test 4: Test text analysis
    print("\n📊 Testing Text Analysis:")
    try:
        sample_text = """
        Microservices architecture is a design pattern that structures an application as a collection 
        of loosely coupled services. This approach enables organizations to develop and deploy 
        applications more efficiently. Each service is self-contained and communicates via 
        well-defined APIs.
        """
        
        analysis_result = await registry.execute_tool("text_analysis", {
            "text": sample_text,
            "analysis_type": "full"
        })
        
        if analysis_result['success']:
            print(f"Word count: {analysis_result['word_count']}")
            print(f"Sentence count: {analysis_result['sentence_count']}")
            print(f"Average words per sentence: {analysis_result['avg_words_per_sentence']}")
            print(f"Top words: {analysis_result['top_words'][:3]}")
        else:
            print(f"Text analysis failed: {analysis_result.get('error')}")
    except Exception as e:
        print(f"Text analysis test failed: {e}")
    
    # Test 5: Test data storage
    print("\n💾 Testing Data Storage:")
    try:
        # Store data
        store_result = await registry.execute_tool("data_storage", {
            "action": "store",
            "key": "test_key",
            "data": {"message": "Hello from tool registry test!", "timestamp": "2025-06-28"},
            "filename": "test_storage"
        })
        print(f"Data store success: {store_result['success']}")
        
        # Retrieve data
        retrieve_result = await registry.execute_tool("data_storage", {
            "action": "retrieve",
            "key": "test_key",
            "filename": "test_storage"
        })
        print(f"Data retrieve success: {retrieve_result['success']}")
        if retrieve_result['success']:
            print(f"Retrieved data: {retrieve_result['data']}")
    except Exception as e:
        print(f"Data storage test failed: {e}")
    
    # Test 6: Test knowledge base
    print("\n🧠 Testing Knowledge Base:")
    try:
        # Store knowledge
        kb_store_result = await registry.execute_tool("knowledge_base", {
            "action": "store",
            "topic": "Microservices Benefits",
            "content": "Microservices provide scalability, flexibility, and technology diversity. They enable teams to work independently and deploy services separately.",
            "tags": ["architecture", "microservices", "scalability"]
        })
        print(f"Knowledge store success: {kb_store_result['success']}")
        
        # Search knowledge
        kb_search_result = await registry.execute_tool("knowledge_base", {
            "action": "search",
            "content": "scalability"
        })
        print(f"Knowledge search success: {kb_search_result['success']}")
        if kb_search_result['success']:
            print(f"Search matches: {kb_search_result['total_matches']}")
    except Exception as e:
        print(f"Knowledge base test failed: {e}")
    
    # Test 7: Test calculator
    print("\n🧮 Testing Calculator:")
    try:
        calc_result = await registry.execute_tool("calculator", {
            "expression": "2 + 3 * 4"
        })
        print(f"Calculator success: {calc_result['success']}")
        if calc_result['success']:
            print(f"Result: {calc_result['expression']} = {calc_result['result']}")
    except Exception as e:
        print(f"Calculator test failed: {e}")
    
    print("\n✅ Tool registry testing completed!")
    
    # Cleanup
    try:
        import os
        if os.path.exists("test_output.txt"):
            os.remove("test_output.txt")
        if os.path.exists("agent_data/test_storage.json"):
            os.remove("agent_data/test_storage.json")
        if os.path.exists("knowledge_base.json"):
            os.remove("knowledge_base.json")
        print("🧹 Cleanup completed")
    except:
        pass

if __name__ == "__main__":
    asyncio.run(test_tools())
