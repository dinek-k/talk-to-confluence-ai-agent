# Talk to Confluence

A Streamlit application that allows you to interact with your Confluence workspace using natural language queries.

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```
CONFLUENCE_URL=your_confluence_url
CONFLUENCE_USERNAME=your_username
CONFLUENCE_API_TOKEN=your_api_token
OPENAI_API_KEY=your_openai_api_key
```

To get your Confluence API token:
1. Log in to Atlassian account settings
2. Navigate to Security > API tokens
3. Create and copy your API token

## Running the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

## Features

- Connect to any Confluence space
- Ask questions about your Confluence content
- Natural language interaction with your documentation 