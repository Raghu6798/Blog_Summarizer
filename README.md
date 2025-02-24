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

## Strategies Implemented before  : 

- **Iterative Refinement**: Implements advanced summarization strategies to refine content.
- ![download (1)](https://github.com/user-attachments/assets/ffe2314b-93f2-4a5c-896a-87191eb9a3ce)
  Iterative refinement represents one strategy for summarizing long texts. The strategy is as follows:

Split a text into smaller documents;
-Summarize the first document;
-Refine or update the result based on the next document;
-Repeat through the sequence of documents until finished.
-It is especially effective when understanding of a sub-document depends on prior context-- for instance, when summarizing a novel or body of text with an inherent sequence.



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
  - `langgraph` (Stateful Pipeline for the dynamic flow of state variables)
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
1. **Search**: Fetches URLs related to the query.  ![images](https://github.com/user-attachments/assets/8294e07f-8877-4236-83b3-a8bb3bf21385)
2. **Crawling**: Extracts article content from the retrieved URLs.  ![hero](https://github.com/user-attachments/assets/4648f278-4c6c-4156-89b6-c37f5874976b)
3. **Classification**: Categorizes articles into sub-topics.
4. **Summarization**: Generates structured summaries.
5. **SEO Optimization**: Improves content for better search visibility.
6. **Image Generation**: Creates images using AI.   ![images](https://github.com/user-attachments/assets/a20a2076-c0c6-4cda-b4e9-c4f09109fc8d)
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

![Screenshot 2025-02-24 135309](https://github.com/user-attachments/assets/1f100da4-9b41-436f-bfeb-11be25fe6ba1)

## Directory Structure
```
AI_Blog_Generator/
│── images/          # AI-generated images
│── outputs/         # Final blog content
│── ai_blog_generator.py  # Main script
│── README.md        # Documentation
│── .env             # API keys (not included in repo)
```

## Leveraged Langgraph's Stateful Pipelines with the below schema : 

``` python
class GraphState(BaseModel):
    query: str
    search_results: Optional[List[Dict[str, str]]] = None
    crawled_content: Optional[Dict[str, str]] = (
        None  # 🔹 New field for extracted Firecrawl content
    )
    classified_articles: Optional[Dict[str, List[str]]] = None
    summaries: Optional[Dict[str, str]] = {}
    seo_optimized_content: Optional[Dict[str, str]] = None
    image_paths: Optional[Dict[str, str]] = None

```

![download](https://github.com/user-attachments/assets/78210597-6631-4658-a997-8b277e745b55)



## Future Improvements
- Improve SEO scoring with advanced analytics.
- Automate blog publishing to platforms like WordPress.
- Implement a user-friendly frontend interface.

## License
This project is open-source under the MIT License.

## Author
Developed by Raghu Nandan Erukulla - Student of Indian Institute of Information Technology Nagpur  
Ayushman Singh - Student of Indian Institute of Information Technology Nagpur
