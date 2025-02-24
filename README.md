# Blog_Summarizer

## Overview
This AI-powered blog summarizer automates the process of researching, classifying, summarizing, optimizing, and illustrating blog articles. It leverages multiple AI models and APIs to extract relevant content from the web, analyze it, generate SEO-friendly summaries, and even create images for blog posts.

## Features
- **Web Search**: Uses Tavily to fetch search results.
- **Content Crawling**: Uses Firecrawl to extract structured content from URLs.
- **Article Classification**: Categorizes articles into relevant sub-topics.
- **Summarization**: Generates structured summaries of articles.
- **SEO Optimization**: Enhances content for search engines.
- **AI-Generated Images**: Creates relevant images for blog content.
- **Display & Export**: Presents results in a structured markdown format.
- **Blogger API Integration**: Publishes the generated summary to Blogger automatically.

## Technologies Used
- **Open Source LLMs**:
  - Gemini-2.0-flash (Google Generative AI)
  - Llama-3.1-8b-instant (Groq)
  - Llama-3.3-70b (Cerebras)
- **Libraries & Tools**:
  - `together`
  - `langchain`
  - `pydantic`
  - `bs4` (BeautifulSoup)
  - `langchain` (LLM Integration & ChatPromptTemplate)
  - 'langgraph` (Stateful Pipeline for the dynamic flow of state variables)
  - `google-auth`
  - `firecrawl`
  - `PIL` (for image handling)
- **APIs Used**:
  - Tavily Search API 
  - Firecrawl API 🔥
  - Google Generative AI 
  - Cerebras AI
  - Together AI
  - Blogger API

## Installation
Ensure you have the required dependencies installed:
```sh
pip install together langchain-groq langchain-google-genai bs4 google-auth google-auth-oauthlib googleapiclient langchain-community pydantic python-dotenv PIL langchain_cerebras langgraph
```

## Environment Variables
Before running the script, set up the following API keys:
```sh
export TAVILY_API_KEY=your_api_key
export GROQ_API_KEY=your_api_key
export GOOGLE_API_KEY=your_api_key
export TOGETHER_API_KEY=your_api_key
export CEREBRAS_API_KEY=your_api_key
export FIRECRAWL_API_KEY=your_api_key
export BLOGGER_API_KEY=your_api_key
```

## Workflow
1. **Search**: Fetches URLs related to the query.
2. **Crawling**: Extracts article content from the retrieved URLs.
3. **Classification**: Categorizes articles into sub-topics.
4. **Summarization**: Generates structured summaries.
5. **SEO Optimization**: Improves content for better search visibility.
6. **Image Generation**: Creates images using AI.
7. **Display Results**: Presents content in markdown format.
8. **Publishing**: Automatically publishes the generated summary to Blogger via the Blogger API.

## Usage
Run the main script to start the AI blog generator:
```sh
python ai_blog_generator.py
```

## Output
- Structured blog content
- AI-generated images
- Markdown-ready text for publishing
- Automated publishing to Blogger

## Directory Structure
```
AI_Blog_Generator/
│── images/          # AI-generated images
│── outputs/         # Final blog content
│── ai_blog_generator.py  # Main script
│── README.md        # Documentation
│── .env             # API keys (not included in repo)
```

## Future Improvements
- Improve SEO scoring with advanced analytics.
- Automate blog publishing to platforms like WordPress.
- Implement a user-friendly frontend interface.

## License
This project is open-source under the MIT License.

## Author
Developed by Raghu Nandan Erukulla - Student of Indian Institute of Information Technology Nagpur  
Ayushman Singh - Student of Indian Institute of Information Technology Nagpur
