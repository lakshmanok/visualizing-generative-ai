from dotenv import load_dotenv
import os
from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

# Load key into the environment
load_dotenv("../keys.env")
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# let's create all 4 LLMs
llms = [
   # OpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")),
    GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY")),
    ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY")),
    # ChatAnthropic(model="claude-3-sonnet", api_key=os.getenv("ANTHROPIC_API_KEY"))
]

prompt = """
Categorize the restaurant review below into one of three categories based on the sentiment of the review: Negative, Neutral, Positive and identify whether the review is discussing food, service, and/or atmosphere.

**Review**:
A real gem of Cantonese cooking. Food was really hot. You can taste the "wok hei" in the food. Service was friendly and quick. Definitely coming again.
"""

for llm in llms:
    print("\n\n\n***", llm, "***\n")
    response = llm.invoke(prompt)
    print(type(response))
    print(response)
