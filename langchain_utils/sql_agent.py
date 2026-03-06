# langchain_utils/sql_agent.py
import logging
from typing import Dict, Any, Optional, List
import sqlite3
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from langchain_core.tools import tool
from llm_setup import get_llm
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# MOVIES DATABASE FUNCTIONS
# ============================================================

def get_database_schema(db_path: str) -> str:
    """Get movies database schema (fruit-style)"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(movies_metadata)")
    columns = cursor.fetchall()

    cursor.execute("SELECT * FROM movies_metadata LIMIT 3")
    samples = cursor.fetchall()

    conn.close()

    schema = "Table: movies_metadata\n\nColumns:\n"
    for col in columns:
        schema += f"- {col[1]} ({col[2]})\n"
    schema += "\nSample rows:\n"
    for row in samples:
        schema += f"- {row}\n"
    return schema

# ============================================================
# CUSTOM TOOLS
# ============================================================

def create_movies_tools(db_path: str):
    """Create fruit-style @tool functions for movies"""
    
    @tool
    def get_schema() -> str:
        """Get the movies database schema to understand table structure."""
        schema = get_database_schema(db_path)
        print("\n📋 Schema Tool: Providing database schema")
        return schema

    @tool
    def execute_sql(sql_query: str) -> str:
        """
        Execute a SQL query on the movies database.
        Input should be a valid SQL SELECT statement.
        
        IMPORTANT: Only SELECT queries are allowed (no INSERT, UPDATE, DELETE).
        """
        print(f"\n💾 SQL Execution Tool: Running query")
        print(f"   Query: {sql_query[:100]}...")
        
        sql_upper = sql_query.upper().strip()
        if not sql_upper.startswith('SELECT'):
            return "Error: Only SELECT queries are allowed for safety."
        
        if any(keyword in sql_upper for keyword in ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER']):
            return "Error: Modification queries are not allowed."
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            column_names = [description[0] for description in cursor.description]
            
            conn.close()
            
            if not results:
                return "Query executed successfully but returned no results."
            
            output = f"Columns: {', '.join(column_names)}\n\n"
            output += f"Results ({len(results)} rows):\n"
            
            for row in results:
                output += f"  {row}\n"
            
            print(f"   ✓ Returned {len(results)} rows")
            return output
        
        except sqlite3.Error as e:
            error_msg = f"SQL Error: {str(e)}"
            print(f"   ❌ {error_msg}")
            return error_msg
    return [get_schema, execute_sql]

# ============================================================
# MOVIE SQL AGENT CLASS 
# ============================================================

class MovieSQLAgent:
    """ReAct SQL Agent for natural language database queries - FRUIT-STYLE TOOLS"""
    
    def __init__(self, db_path: str, hf_api_key: Optional[str] = None):
        """
        Initialize SQL Agent with fruit-style @tools
        
        Args:
            db_path: Path to SQLite database
            hf_api_key: HuggingFace API key
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize fruit-style tools
        self.tools = create_movies_tools(db_path)
        
        # Initialize ReAct agent with custom tools
        if hf_api_key:
            self.logger.info("Initializing MovieSQLAgent with fruit-style tools...")
            self.llm = get_llm(api_key=hf_api_key, model_type='qwen')
            
            # Create agent with FRUIT-STYLE custom tools (not SQLDatabaseToolkit)
            self.agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt="""You are a SQL expert assistant with access to a movies database. Answer the user's question by generating and executing SQL queries.
When answering questions about the database:
1. FIRST call get_schema() to see the database structure
2. THEN write a SQL SELECT query to answer the question
3. Call execute_sql() with your generated query
4. Interpret the results and answer in natural language

IMPORTANT SQL RULES:
- Always use SELECT queries only
- Use proper SQL syntax (SQLite)
- Handle NULL values appropriately
- Use LIMIT when appropriate
- Join tables if needed (though we only have one table)

Example workflow:
User: "Suggest me the top 5 highest rated movies"
1. Call get_schema() to see columns
2. Generate: SELECT title, vote_average FROM movies_metadata ORDER BY vote_average DESC LIMIT 5;
3. Call execute_sql() with that query
4. Format results naturally

You have access to the following tools, you can call them as needed to get information or execute queries:
get_schema() - returns the database schema
execute_sql(sql_query) - executes a SQL query and returns results
"""
            )
            
            self.is_ready = True
            self.logger.info("MovieSQLAgent initialized with fruit-style tools")
        else:
            self.logger.warning("No API key - SQL Agent unavailable")
            self.llm = None
            self.agent = None
            self.is_ready = False
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute natural language query using fruit-style agent
        (Exact same interface as before)
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
            
            # Run agent with fruit-style tools
            response = self.agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            
            # Extract output (same as before)
            final_message = response["messages"][-1]
            output = final_message.content
            
            # Extract SQL from tool calls (same logic)
            sql_query = None
            if hasattr(response, 'intermediate_steps') and response.intermediate_steps:
                for step in response.intermediate_steps:
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
            error_msg = str(e)
            if "parsing" in error_msg.lower():
                error_msg = "The AI model had trouble generating a proper response. Try rephrasing your question."
            
            return {
                'success': False,
                'result': None,
                'sql_query': None,
                'error': error_msg
            }
    
    # All other methods remain EXACTLY THE SAME
    def get_sample_queries(self) -> List[str]:
        return [
            "Show me the top 10 highest rated movies",
            "What are the 5 longest movies?",
            "Find movies released in 2015",
            "Which movies have more than 1000 votes?",
            "Show me Action movies",
            "What are the most popular movies?"
        ]
    
    def get_table_info(self) -> str:
        return get_database_schema(self.db_path)
    
    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """Direct SQL execution (unchanged)"""
        if not sql_query.strip().upper().startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed")
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return df
        except Exception as e:
            self.logger.error(f"SQL execution failed: {e}")
            raise
    
    def list_tables(self) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

if __name__ == "__main__":
    # Example usage
    db_path = "data/moviesdb_etl_upd.db"
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    agent = MovieSQLAgent(db_path=db_path, hf_api_key=api_key)
    
    if agent.is_ready:
        # question = "What are the top 5 highest rated movies?"
        question = "What are the top 5 longest movies?"
        response = agent.query(question)
        print(response)
