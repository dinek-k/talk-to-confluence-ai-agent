from typing import List, Optional, Dict
from atlassian import Confluence
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import chromadb
import tempfile

load_dotenv()

class ConfluenceAgent:
    def __init__(self):
        self.confluence = None
        self.space_key = None
        self.llm = ChatOpenAI(temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store_stats = {
            'total_pages': 0,
            'total_chunks': 0,
            'last_updated': None,
            'status': 'Not initialized'
        }
        
    def connect_to_confluence(self, space_key: str) -> bool:
        """Connect to Confluence with the specified space key."""
        try:
            self.confluence = Confluence(
                url=os.getenv('CONFLUENCE_URL'),
                username=os.getenv('CONFLUENCE_USERNAME'),
                password=os.getenv('CONFLUENCE_API_TOKEN')
            )
            self.space_key = space_key
            # Test the connection by trying to get space info
            self.confluence.get_space(space_key)
            # Initialize vector store
            self.vector_store_stats['status'] = 'Initializing...'
            self.initialize_vector_store()
            return True
        except Exception as e:
            print(f"Error connecting to Confluence: {str(e)}")
            self.vector_store_stats['status'] = f'Error: {str(e)}'
            return False

    def clean_html_content(self, html_content: str) -> str:
        """Clean HTML content and extract text."""
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def get_page_content(self, page_id: str) -> str:
        """Get the content of a specific page."""
        try:
            page = self.confluence.get_page_by_id(page_id, expand='body.storage')
            content = page['body']['storage']['value']
            return self.clean_html_content(content)
        except Exception as e:
            return f"Error retrieving page content: {str(e)}"

    def initialize_vector_store(self):
        """Initialize the vector store with all pages from the space."""
        try:
            # Create a temporary directory for ChromaDB
            temp_dir = tempfile.mkdtemp()
            
            # Get all pages from the space
            all_pages = self.confluence.get_all_pages_from_space(
                self.space_key,
                start=0,
                limit=500,  # Adjust this limit based on your space size
                status='current',
                expand='body.storage'
            )

            documents = []
            metadatas = []
            
            for page in all_pages:
                content = self.clean_html_content(page['body']['storage']['value'])
                # Split content into chunks
                texts = self.text_splitter.split_text(content)
                
                # Add each chunk as a document with metadata
                for text in texts:
                    documents.append(text)
                    metadatas.append({
                        'page_id': page['id'],
                        'title': page['title'],
                        'url': f"{os.getenv('CONFLUENCE_URL')}/wiki/spaces/{self.space_key}/pages/{page['id']}"
                    })

            # Create vector store
            self.vector_store = Chroma.from_texts(
                texts=documents,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory=temp_dir
            )
            
            # Update stats
            from datetime import datetime
            self.vector_store_stats.update({
                'total_pages': len(all_pages),
                'total_chunks': len(documents),
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'Ready'
            })
            
            print(f"Vector store initialized with {len(documents)} chunks from {len(all_pages)} pages")
            
        except Exception as e:
            error_msg = f"Error initializing vector store: {str(e)}"
            print(error_msg)
            self.vector_store_stats['status'] = f'Error: {error_msg}'

    def get_vector_store_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return self.vector_store_stats

    def search_vector_store(self, query: str, k: int = 5) -> List[dict]:
        """Search the vector store for relevant content."""
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        formatted_results = []
        
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': score
            })
            
        return formatted_results

    def setup_agent(self) -> AgentExecutor:
        """Set up the LangChain agent with tools."""
        tools = [
            Tool(
                name="search_confluence",
                func=self.search_vector_store,
                description="Search for relevant content in the Confluence space using semantic search. Returns the most relevant content chunks with their metadata."
            ),
            Tool(
                name="get_page_content",
                func=self.get_page_content,
                description="Get the full content of a specific Confluence page. Input should be the page ID."
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that helps users find information in their Confluence workspace.
            Use the vector store search tool to find relevant content, and the page content tool to get more details if needed.
            Always provide clear and concise responses based on the actual content found in Confluence.
            Include relevant page titles and URLs in your responses when appropriate."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_functions_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def query(self, user_input: str) -> str:
        """Process a user query using the agent."""
        if not self.confluence or not self.space_key:
            return "Please connect to a Confluence space first."
        
        try:
            agent_executor = self.setup_agent()
            response = agent_executor.invoke({"input": user_input})
            return response["output"]
        except Exception as e:
            return f"Error processing query: {str(e)}" 