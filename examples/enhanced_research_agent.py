"""
Enhanced Research Agent Example

This example demonstrates an improved research agent that actively uses tools
and generates comprehensive research reports with file output.
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.agent import Agent, AgentGoal, AgentCapability
from src.core.orchestrator import AgentOrchestrator
from src.integration.ollama_client import OllamaIntegrationLayer, OllamaResponse, OllamaToolCall
from src.tools.tool_registry import ToolRegistry
from src.memory.memory_manager import MemoryManager


class EnhancedResearchAgent:
    """Enhanced research agent that actively uses tools and generates reports."""
    
    def __init__(self, orchestrator, agent_id):
        self.orchestrator = orchestrator
        self.agent_id = agent_id
        self.research_data = []
        
    async def conduct_research(self, topic, focus_areas=None):
        """Conduct comprehensive research on a topic."""
        print(f"🔍 Starting research on: {topic}")
        
        # Get the tool registry
        tool_registry = self.orchestrator.tool_registry
        
        # Perform multiple searches for comprehensive coverage
        search_queries = [
            f"{topic} current trends 2024",
            f"{topic} latest developments",
            f"{topic} challenges and opportunities"
        ]
        
        if focus_areas:
            for area in focus_areas:
                search_queries.append(f"{topic} {area}")
        
        # Conduct searches
        for query in search_queries:
            try:
                print(f"  🔎 Searching: {query}")
                result = await tool_registry.execute_tool("web_search", {
                    "query": query,
                    "max_results": 3
                })
                
                if result['success']:
                    self.research_data.append({
                        'query': query,
                        'results': result['results'],
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"    ✅ Found {len(result['results'])} results")
                else:
                    print(f"    ❌ Search failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"    ❌ Search error: {e}")
        
        # Analyze the collected data
        await self.analyze_research_data()
        
        # Generate and save report
        report = await self.generate_report(topic)
        await self.save_report(report, topic)
        
        return report
    
    async def analyze_research_data(self):
        """Analyze the collected research data."""
        print("📊 Analyzing research data...")
        
        tool_registry = self.orchestrator.tool_registry
        
        # Combine all research content for analysis
        all_content = ""
        for data in self.research_data:
            for result in data['results']:
                all_content += f"{result['title']}: {result['snippet']} "
        
        if all_content:
            try:
                analysis = await tool_registry.execute_tool("text_analysis", {
                    "text": all_content,
                    "analysis_type": "full"
                })
                
                if analysis['success']:
                    print(f"  📝 Analyzed {analysis['word_count']} words")
                    print(f"  🔤 Key terms: {[word[0] for word in analysis['top_words'][:5]]}")
                else:
                    print("  ❌ Analysis failed")
                    
            except Exception as e:
                print(f"  ❌ Analysis error: {e}")
    
    async def generate_report(self, topic):
        """Generate a comprehensive research report."""
        print("📄 Generating research report...")
        
        report_sections = []
        
        # Executive Summary
        report_sections.append("# Research Report: " + topic.title())
        report_sections.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("")
        report_sections.append("## Executive Summary")
        report_sections.append(f"This report presents comprehensive research findings on {topic}. ")
        report_sections.append("The research encompasses current trends, latest developments, and key challenges in the field.")
        report_sections.append("")
        
        # Research Methodology
        report_sections.append("## Research Methodology")
        report_sections.append(f"- Conducted {len(self.research_data)} targeted web searches")
        report_sections.append("- Analyzed multiple sources for comprehensive coverage")
        report_sections.append("- Synthesized findings to identify key trends and insights")
        report_sections.append("")
        
        # Key Findings
        report_sections.append("## Key Findings")
        
        finding_counter = 1
        for data in self.research_data:
            for result in data['results']:
                report_sections.append(f"### {finding_counter}. {result['title']}")
                report_sections.append(f"**Source:** {result.get('url', 'Internal Research')}")
                report_sections.append(f"**Summary:** {result['snippet']}")
                report_sections.append("")
                finding_counter += 1
        
        # Trend Analysis
        report_sections.append("## Trend Analysis")
        report_sections.append("Based on the research conducted, several key trends emerge:")
        
        # Extract common themes from the research
        all_titles = [result['title'] for data in self.research_data for result in data['results']]
        common_themes = self._extract_themes(all_titles)
        
        for i, theme in enumerate(common_themes, 1):
            report_sections.append(f"{i}. **{theme}**: Consistently mentioned across multiple sources")
        
        report_sections.append("")
        
        # Challenges and Opportunities
        report_sections.append("## Challenges and Opportunities")
        report_sections.append("### Challenges:")
        report_sections.append("- Rapid pace of technological change requiring constant adaptation")
        report_sections.append("- Need for skilled professionals and continuous learning")
        report_sections.append("- Balancing innovation with ethical considerations")
        report_sections.append("")
        report_sections.append("### Opportunities:")
        report_sections.append("- Significant potential for growth and innovation")
        report_sections.append("- Cross-industry applications and integration possibilities")
        report_sections.append("- Development of new business models and services")
        report_sections.append("")
        
        # Recommendations
        report_sections.append("## Recommendations")
        report_sections.append("1. Stay informed about emerging trends and technologies")
        report_sections.append("2. Invest in skill development and training")
        report_sections.append("3. Foster collaboration between different stakeholders")
        report_sections.append("4. Prioritize ethical and responsible development")
        report_sections.append("5. Explore practical applications and implementation strategies")
        report_sections.append("")
        
        # Research Data Appendix
        report_sections.append("## Appendix: Detailed Research Data")
        for i, data in enumerate(self.research_data, 1):
            report_sections.append(f"### Search {i}: {data['query']}")
            for j, result in enumerate(data['results'], 1):
                report_sections.append(f"**Result {j}:**")
                report_sections.append(f"- Title: {result['title']}")
                report_sections.append(f"- URL: {result.get('url', 'N/A')}")
                report_sections.append(f"- Type: {result.get('type', 'Unknown')}")
                report_sections.append(f"- Content: {result['snippet']}")
                report_sections.append("")
        
        return "\n".join(report_sections)
    
    def _extract_themes(self, titles):
        """Extract common themes from research titles."""
        themes = []
        common_words = ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'technology', 
                       'development', 'trends', 'innovation', 'future', 'applications']
        
        # Simple theme extraction based on common words
        for word in common_words:
            if any(word.lower() in title.lower() for title in titles):
                themes.append(word.title())
        
        return themes[:5]  # Return top 5 themes
    
    async def save_report(self, report, topic):
        """Save the research report to a file."""
        tool_registry = self.orchestrator.tool_registry
        
        # Clean topic name for filename
        clean_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"research_report_{clean_topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            result = await tool_registry.execute_tool("file_write", {
                "filename": filename,
                "content": report,
                "create_dirs": True
            })
            
            if result['success']:
                print(f"💾 Report saved: {result['filename']}")
                print(f"📊 Report size: {result['content_length']} characters")
                return result['filename']
            else:
                print(f"❌ Failed to save report: {result.get('error')}")
                return None
                
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return None


async def main():
    """Main function to demonstrate the enhanced research agent."""
    print("🔬 Enhanced Research Agent Example")
    print("=" * 50)
    
    # Initialize components
    print("Initializing framework components...")
    
    ollama_integration = OllamaIntegrationLayer()
    await ollama_integration.initialize()
    
    tool_registry = ToolRegistry()
    memory_manager = MemoryManager(db_path=":memory:")
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        ollama_integration=ollama_integration,
        tool_registry=tool_registry,
        memory_manager=memory_manager
    )
    
    # Create enhanced research agent
    research_agent = EnhancedResearchAgent(orchestrator, "enhanced_researcher_001")
    
    print("✅ Enhanced research agent initialized")
    
    # Start the orchestrator
    await orchestrator.start()
    
    # Define research parameters
    research_topic = "artificial intelligence trends"
    focus_areas = ["machine learning", "natural language processing", "computer vision", "ethics"]
    
    print(f"\n🎯 Research Topic: {research_topic}")
    print(f"📋 Focus Areas: {', '.join(focus_areas)}")
    
    # Conduct research
    try:
        report = await research_agent.conduct_research(research_topic, focus_areas)
        
        if report:
            print("\n🎉 Research completed successfully!")
            print(f"📄 Report length: {len(report)} characters")
            
            # Display summary statistics
            lines = report.split('\n')
            sections = [line for line in lines if line.startswith('#')]
            print(f"📑 Report sections: {len(sections)}")
            print(f"📊 Research data points: {len(research_agent.research_data)}")
            
        else:
            print("❌ Research failed or no report generated")
            
    except Exception as e:
        print(f"❌ Research error: {e}")
    
    # Stop the orchestrator
    await orchestrator.stop()
    
    # Final statistics
    print("\n📈 Final Statistics:")
    memory_stats = await memory_manager.get_memory_stats()
    tool_count = tool_registry.get_tool_count()
    
    print(f"Memory entries: {memory_stats['total_memories']}")
    print(f"Available tools: {tool_count}")
    print(f"Research queries executed: {len(research_agent.research_data)}")
    
    print("\n✨ Enhanced research example completed!")


if __name__ == "__main__":
    asyncio.run(main())
