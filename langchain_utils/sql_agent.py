# langchain_utils/sql_agent.py
"""
SQL Agent with ReAct reasoning for natural language database queries
"""

import logging
from typing import Dict, Any, Optional, List
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from .llm_setup import get_llm
import pandas as pd

# Custom SQL Agent Prompt - More explicit for open-source models
SQL_PREFIX = """You are a SQL expert assistant. Answer the user's question by generating and executing SQL queries.

You have access to the following tools:

{tools}

The database contains ONE table called 'movies_metadata' with these columns:
- id (INTEGER): Movie ID
- title (TEXT): Movie title  
- genres (TEXT): Comma-separated genres
- overview (TEXT): Movie description
- vote_average (REAL): Rating (0-10)
- vote_count (INTEGER): Number of votes
- release_date (TEXT): Release date
- release_year (INTEGER): Release year
- runtime (INTEGER): Runtime in minutes
- popularity (REAL): Popularity score
- original_language (TEXT): Language code

Use this EXACT format for your response:

Question: the input question
Thought: think about what to do
Action: the action to take (must be one of [{tool_names}])
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the question

IMPORTANT RULES:
1. Always write Action and Action Input on separate lines
2. Generate clean SQL SELECT queries only
3. Use ORDER BY and LIMIT for top results
4. Check if the query returned results before answering

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


class MovieSQLAgent:
    """ReAct SQL Agent for natural language database queries"""
    
    def __init__(self, db_path: str, hf_api_key: Optional[str] = None):
        """
        Initialize SQL Agent with ReAct reasoning
        
        Args:
            db_path: Path to SQLite database
            hf_api_key: HuggingFace API key
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self.logger.info(f"Connecting to database: {db_path}")
        self.db = SQLDatabase.from_uri(
            f"sqlite:///{db_path}", 
            include_tables=['movies_metadata']
        )
        
        # Initialize ReAct agent
        if hf_api_key:
            self.logger.info("Initializing ReAct SQL Agent...")
            self.llm = get_llm(api_key=hf_api_key, model_type='qwen')  # Using Qwen2.5-72B-Instruct
            
            # Create toolkit
            self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            
            # Create agent with custom prompt
            # self.agent = create_sql_agent(
            #     llm=self.llm,
            #     toolkit=self.toolkit,
            #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            #     prefix=SQL_PREFIX,
            #     verbose=True,
            #     handle_parsing_errors=True,
            #     max_iterations=10,
            #     max_execution_time=60,
            #     return_intermediate_steps=True
            # )
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=self.toolkit,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )

            
            self.is_ready = True
            self.logger.info("ReAct SQL Agent initialized")
        else:
            self.logger.warning("No API key - SQL Agent unavailable")
            self.llm = None
            self.toolkit = None
            self.agent = None
            self.is_ready = False
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute natural language query using ReAct agent
        
        Args:
            question: Natural language question
            
        Returns:
            Dict with success, result, sql_query, error
        """
        if not self.is_ready:
            return {
                'success': False,
                'result': None,
                'sql_query': None,
                'error': 'SQL Agent not initialized. Add HUGGINGFACE_API_KEY to .env'
            }
        
        try:
            self.logger.info(f"Query: {question}")
            
            # Run ReAct agent
            response = self.agent.invoke({"input": question})
            
            # Extract output
            output = response.get('output', '')
            
            # Try to extract SQL from intermediate steps
            sql_query = None
            if 'intermediate_steps' in response:
                for step in response['intermediate_steps']:
                    if len(step) >= 2:
                        action, observation = step
                        if hasattr(action, 'tool_input'):
                            tool_input = action.tool_input
                            if isinstance(tool_input, str) and 'SELECT' in tool_input.upper():
                                sql_query = tool_input
                                break
            
            self.logger.info("Query executed successfully")
            
            return {
                'success': True,
                'result': output,
                'sql_query': sql_query,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"Error: {str(e)}", exc_info=True)
            
            # Try to extract useful error info
            error_msg = str(e)
            if "parsing" in error_msg.lower():
                error_msg = "The AI model had trouble generating a proper response. Try rephrasing your question."
            
            return {
                'success': False,
                'result': None,
                'sql_query': None,
                'error': error_msg
            }
    
    def get_sample_queries(self) -> List[str]:
        """Sample queries for user guidance"""
        return [
            "Show me the top 10 highest rated movies",
            "What are the 5 longest movies?",
            "Find movies released in 2015",
            "Which movies have more than 1000 votes?",
            "Show me Action movies",
            "What are the most popular movies?"
        ]
    
    def get_table_info(self) -> str:
        """Get database schema info"""
        return self.db.get_table_info()
    
    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query directly (for advanced users)
        
        Args:
            sql_query: SQL query string
            
        Returns:
            DataFrame with results
        """
        # Safety check: only allow SELECT queries
        if not sql_query.strip().upper().startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed")
        
        try:
            result = self.db.run(sql_query)
            # Convert to DataFrame if possible
            if isinstance(result, str):
                return pd.DataFrame({'result': [result]})
            return pd.DataFrame(result)
        except Exception as e:
            self.logger.error(f"SQL execution failed: {e}")
            raise
    
    def get_schema_info(self) -> str:
        """Get database schema information"""
        return self.db.get_table_info()
    
    def list_tables(self) -> list:
        """List all tables in the database"""
        return self.db.get_usable_table_names()
    
    def get_sample_queries(self) -> list:
        """Get sample queries for reference"""
        return [
            "Show me the top 10 highest rated movies",
            "Find all action movies released in 2015",
            "What are the longest movies in the database?",
            "Show me sci-fi movies with more than 1000 votes",
            "List movies directed by Christopher Nolan",
            "Find movies with 'space' in the title",
            "Show me the most popular movies from 2010",
            "What genres are most common in the database?"
        ]
