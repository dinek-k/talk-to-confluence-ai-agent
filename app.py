import streamlit as st
from confluence_agent import ConfluenceAgent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = ConfluenceAgent()
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("Talk to Confluence")

# Sidebar for Confluence connection and stats
with st.sidebar:
    st.header("Connection Settings")
    
    if not st.session_state.connected:
        space_key = st.text_input("Enter Confluence Space Key")
        if st.button("Connect"):
            if not all([os.getenv('CONFLUENCE_URL'), 
                       os.getenv('CONFLUENCE_USERNAME'),
                       os.getenv('CONFLUENCE_API_TOKEN'),
                       os.getenv('OPENAI_API_KEY')]):
                st.error("Please set all required environment variables in .env file")
            elif space_key:
                with st.spinner("Connecting to Confluence..."):
                    if st.session_state.agent.connect_to_confluence(space_key):
                        st.session_state.connected = True
                        st.success("Connected successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to connect. Please check your credentials and space key.")
            else:
                st.warning("Please enter a space key")
    else:
        st.success("Connected to Confluence")
        if st.button("Disconnect"):
            st.session_state.connected = False
            st.session_state.agent = ConfluenceAgent()
            st.rerun()
        
        # Display vector store statistics
        st.header("Vector Store Stats")
        stats = st.session_state.agent.get_vector_store_stats()
        
        st.metric("Total Pages", stats['total_pages'])
        st.metric("Total Content Chunks", stats['total_chunks'])
        
        status_color = {
            'Ready': 'green',
            'Initializing...': 'blue',
        }
        status = stats['status']
        color = status_color.get(status, 'red')
        st.markdown(f"**Status:** :{color}[{status}]")
        
        if stats['last_updated']:
            st.markdown(f"**Last Updated:** {stats['last_updated']}")

# Main chat interface
if st.session_state.connected:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your Confluence space"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please connect to a Confluence space using the sidebar to start chatting.") 