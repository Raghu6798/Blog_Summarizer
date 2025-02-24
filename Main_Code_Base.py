import os
from together import Together
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from bs4 import BeautifulSoup
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain.prompts import ChatPromptTemplate
from langchain_cerebras import ChatCerebras
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_community.tools import TavilySearchResults
from IPython.display import display, Markdown
from typing import List, Dict, Optional
import base64
from PIL import Image
from io import BytesIO
import re
import time

load_dotenv()


os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
os.environ["CEREBRAS_API_KEY"] = os.getenv("CEREBRAS_API_KEY")

Gemini_2 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    max_output_tokens=1024,
)

Llama31 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_tokens=4096,
)
Llama33 = ChatCerebras(
    model="llama-3.3-70b",
    temperature=0.6,
    max_tokens=4096,
    api_key=os.getenv("CEREBRAS_API_KEY"),
)


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


blog_graph = StateGraph(state_schema=GraphState)


# 🔹 Step 1: Search Node - Fetch URLs
def search_node(state: GraphState) -> GraphState:
    """
    Performs a web search using Tavily and extracts valid titles, content, and URLs.
    """
    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True,
    )

    search_results = tavily_tool.run(state.query)

    formatted_results = []

    for item in search_results:
        url = item.get("url", "#").strip()
        content = item.get("content", "").strip()

        if len(content) > 10:
            first_sentence = content.split(".")[0]
            title = (
                first_sentence[:100] if len(first_sentence) > 5 else "Untitled Article"
            )
        else:
            title = "Untitled Article"

        if len(content) < 50:
            continue

        formatted_results.append({"title": title, "url": url, "content": content})

    state.search_results = formatted_results

    if not state.search_results:
        print("⚠️ No valid search results retrieved!")

    return state


# 🔹 Step 2: Crawl Node - Extract Content with Firecrawl
def crawl_node(state: GraphState) -> GraphState:
    """
    Uses Firecrawl to extract structured content from the search results' URLs.
    """
    crawled_content = {}

    for result in state.search_results:
        url = result["url"]
        try:
            print(f"🕵️ Crawling: {url}")
            loader = FireCrawlLoader(
                api_key=os.getenv("FIRECRAWL_API_KEY"), url=url, mode="crawl"
            )
            docs = loader.load()  # Fetch structured content
            crawled_content[url] = (
                docs[0].page_content if docs else "No content extracted"
            )
            time.sleep(5)  # Prevent rate limiting
        except Exception as e:
            print(f"❌ Failed to crawl {url}: {e}")

    state.crawled_content = crawled_content
    return state


# 🔹 Step 3: Classify Articles Node
def classify_node(state: GraphState) -> GraphState:
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    classified_articles = {}

    for url, content in state.crawled_content.items():
        classification_prompt = (
            f"Classify the following article into a sub-topic:\n\n{content[:500]}"
        )
        sub_topic = model.invoke(classification_prompt).content

        if sub_topic not in classified_articles:
            classified_articles[sub_topic] = []
        classified_articles[sub_topic].append(url)

    state.classified_articles = classified_articles
    return state


# 🔹 Step 4: Summarization Node
def summarize_node(state: GraphState) -> GraphState:
    summaries = {}

    # Define a structured, multi-step ChatPromptTemplate
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at analyzing and summarizing long-form content.",
            ),
            (
                "human",
                """Analyze the following content: {{article_content}}

        **Required Analysis:**
        1️⃣ Extract **key insights** and major findings.
        2️⃣ Identify **contradictions and opposing views** (if any).
        3️⃣ Highlight **statistics, trends, or research findings**.
        4️⃣ Summarize in a **neutral, unbiased manner**.

        📌 **Output Format**:
        - Key Insights:
        - Contradictions:
        - Statistics & Trends:
        - Final Summary:
        """,
            ),
        ]
    )

    for sub_topic, urls in state.classified_articles.items():
        articles_content = "\n\n".join(
            [state.crawled_content[url] for url in urls if url in state.crawled_content]
        )
        if not articles_content:
            continue

        prompt = summarization_prompt.format(article_content=articles_content)
        refined_summary = Llama33.invoke(prompt).content
        summaries[sub_topic] = refined_summary

    state.summaries = summaries
    return state


# 🔹 Step 5: SEO Optimization Node
def seo_optimize_node(state: GraphState) -> GraphState:
    model = ChatGoogleGenerativeAI(model="gemini-pro")
    seo_optimized_content = {}

    for sub_topic, summary in state.summaries.items():
        seo_prompt = f"Optimize the following text for SEO:\n\n{summary}"
        optimized_text = model.invoke(seo_prompt).content
        seo_optimized_content[sub_topic] = optimized_text

    state.seo_optimized_content = seo_optimized_content
    return state


# 🔹 Step 6: Image Generation Node
def image_generation_node(state: GraphState) -> GraphState:
    client = Together()

    if state.image_paths is None:
        state.image_paths = {}

    save_dir = "C:\\Users\\Raghu\\Downloads\\Agentic_AI_blog_generator"
    os.makedirs(save_dir, exist_ok=True)

    for topic, content in state.seo_optimized_content.items():
        topic_name = Gemini_2.invoke(
            f"Generate a **single short topic title** (max 5 words) for the following content:\n\n{content}"
        ).content.strip()

        safe_topic_name = re.sub(r"[^\w\-_.]", "_", topic_name)[:30]

        image_prompt = Llama31.invoke(f"""
        Create a **realistic, detailed image prompt** for: {topic_name}.
        - Describe a visual scene, avoiding abstract concepts.
        - Example: If the topic is "AI in Finance", describe a **stock trading floor** with AI assistants.

        **Topic:** {topic_name}
        **Image Prompt:**
        """).content.strip()

        response = client.images.generate(
            prompt=image_prompt,
            model="black-forest-labs/FLUX.1-schnell-Free",
            width=1024,
            height=768,
            steps=1,
            n=1,
            response_format="b64_json",
        )

        image_data = base64.b64decode(response.data[0].b64_json)
        image = Image.open(BytesIO(image_data))
        image_path = os.path.join(save_dir, f"{safe_topic_name}.png")

        try:
            image.save(image_path)
            print(f"✅ Image saved: {image_path}")
        except Exception as e:
            print(f"❌ Error saving image: {e}")

        state.image_paths[topic] = image_path

    return state


# 🔹 Step 7: Display Results Node
def display_results_node(state: GraphState) -> GraphState:
    display(Markdown("## 📌 AI Blog Generation Results\n---"))

    search_results_md = "### 🔎 Search Results\n"
    for idx, result in enumerate(state.search_results, start=1):
        search_results_md += f"- **[{result['title']}]({result['url']})**\n"
    print(display(Markdown(search_results_md)))

    for topic, summary in state.summaries.items():
        if topic in state.image_paths:
            display(Image.open(state.image_paths[topic]))
        display(Markdown(f"### 📖 {topic}\n\n{summary}\n---"))

    for topic, seo_content in state.seo_optimized_content.items():
        if topic in state.image_paths:
            display(Image.open(state.image_paths[topic]))

    display(Markdown("## ✅ AI Blog Summarization Completed 🎉"))


# 🔹 Graph Workflow Definition
blog_gen_graph = StateGraph(state_schema=GraphState)
blog_gen_graph.add_node("search", search_node)
blog_gen_graph.add_node("crawl", crawl_node)
blog_gen_graph.add_node("classify", classify_node)
blog_gen_graph.add_node("summarize", summarize_node)
blog_gen_graph.add_node("seo_optimize", seo_optimize_node)
blog_gen_graph.add_node("image_generation", image_generation_node)
blog_gen_graph.add_node("display", display_results_node)

blog_gen_graph.add_edge(START, "search")
blog_gen_graph.add_edge("search", "crawl")
blog_gen_graph.add_edge("crawl", "classify")
blog_gen_graph.add_edge("classify", "summarize")
blog_gen_graph.add_edge("summarize", "seo_optimize")
blog_gen_graph.add_edge("seo_optimize", "image_generation")
blog_gen_graph.add_edge("image_generation", "display")
blog_gen_graph.add_edge("display", END)

compiled_graph = blog_gen_graph.compile()

initial_state = GraphState(
    query="The Evolution of Human Intelligence: Are We Getting Smarter?"
)
final_state = compiled_graph.invoke(initial_state)


SCOPES = ["https://www.googleapis.com/auth/blogger"]
BLOG_ID = "1171030831243402665"  # Replace with your Blogger blog ID


# Authenticate using credentials.json
def authenticate_google():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds


def generate_title(summary):
    """Generates a concise, engaging title using Llama3.3-70B."""
    prompt = f"Generate a concise, engaging, and SEO-friendly blog post title based on the following summary:\n\n{summary}\n\nTitle:"
    response = Llama33(prompt)
    return response.text.strip()

# Function to publish a blog post with AI-generated summaries and images
def publish_ai_blog(state):
    creds = authenticate_google()
    service = build("blogger", "v3", credentials=creds)

    for topic, summary in state["summaries"].items():
        seo_content = state["seo_optimized_content"].get(
            topic, summary
        )  # Use SEO text if available

        # Format the post content
        post_body = {
            "title": topic,
            "content": f"""
            <h2>{topic}</h2>
            <p>{seo_content}</p>
            """,
        }

        # Publish the post
        response = (
            service.posts()
            .insert(blogId=BLOG_ID, body=post_body, isDraft=False)
            .execute()
        )
        print(f"✅ Blog post published: {response['url']}")


publish_ai_blog(final_state) 

