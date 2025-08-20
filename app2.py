
#Updated code
import os
import streamlit as st
import requests
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

if not GROQ_API_KEY or not UNSPLASH_ACCESS_KEY:
    st.error("Missing GROQ_API_KEY or UNSPLASH_ACCESS_KEY in .env file")
    st.stop()

# Setup LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

# Streamlit UI
st.title("📝 Multi-Agent AI Blog Generator")
st.write("Autonomously generate blogs with AI agents.")

topic = st.text_input("Enter a blog topic:", "Artificial Intelligence in Education")
tone = st.selectbox("Choose a writing tone:", ["Professional", "Casual", "Inspirational", "Technical"])
language = st.selectbox("Choose language:", ["English", "French", "Spanish", "German"])

if st.button("Generate Blog"):
    with st.spinner("Agents are working..."):

        # Define agents
        blog_writer = Agent(role="Blog Writer", goal="Write engaging blogs", backstory="An expert writer.", llm=llm)
        editor = Agent(role="Editor", goal="Polish the content", backstory="Experienced editor.", llm=llm)

        # Define tasks (✅ fixed with expected_output)
        writing_task = Task(
            description=f"Write a {tone} blog on {topic} in {language}.",
            expected_output="A complete, well-structured blog article with headings and paragraphs.",
            agent=blog_writer
        )

        editing_task = Task(
            description="Edit and refine the blog for grammar, clarity, and tone.",
            expected_output="A polished final version of the blog, ready for publishing.",
            agent=editor
        )

        # Create crew
        crew = Crew(
            agents=[blog_writer, editor],
            tasks=[writing_task, editing_task],
            verbose=True
        )

        result = crew.kickoff()
        blog_content = result if isinstance(result, str) else result.raw_output

        st.subheader("✅ Generated Blog")
        st.write(blog_content)

        # Fetch related image from Unsplash
        st.subheader("🖼 Suggested Image")
        url = f"https://api.unsplash.com/photos/random?query={topic}&client_id={UNSPLASH_ACCESS_KEY}"
        response = requests.get(url).json()
        image_url = response.get("urls", {}).get("regular")

        if image_url:
            st.image(image_url, caption=f"Image related to {topic}", use_container_width=True)
        else:
            st.warning("No related image found on Unsplash.")
