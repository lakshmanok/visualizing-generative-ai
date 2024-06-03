from dotenv import load_dotenv
import os
import dspy
from dspy import teleprompt

# Load key into the environment
load_dotenv("keys.env")

# Control the creativity
kwargs = {
    "temperature": 0.1,
    "top_p": 1.0,
    # "top_k": 1,
    # "num_generations": 1
}

# let's create all 4 LLMs
llms = [
    # dspy.OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), **kwargs),
    dspy.Google(model="models/gemini-1.0-pro", api_key=os.getenv("GOOGLE_API_KEY"), **kwargs),
    dspy.GROQ(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"), **kwargs),
    # dspy.Claude(model="claude-3-sonnet", api_key=os.getenv("ANTHROPIC_API_KEY"), **kwargs)
]


class ReviewClassifier(dspy.Module):
    """
    Given a review, specifies the sentiment and identifies the topics discussed.
    """
    def __init__(self):
        super().__init__()
        self.context = """
        Categorize the restaurant review below into one of three categories based on the sentiment of the review: Negative, Neutral, Positive and identify whether the review is discussing food, service, and/or atmosphere.
        """
        self.prog = dspy.Predict("context, review -> sentiment, topics, explanation")

    def forward(self, review):
        answer = self.prog(context=self.context, review=review)
        # let's do some post-processing to convert the output into consistent case, array, etc.
        if "Sentiment: " in answer.sentiment:
            start = answer.sentiment.rindex("Sentiment: ") + len("Sentiment: ")
            answer.sentiment = answer.sentiment[start:].upper()
        else:
            answer.sentiment = answer.sentiment.upper()
        answer.topics = [x.strip().upper() for x in answer.topics.split(",")]
        return answer


for llm in llms:
    with dspy.settings.context(lm=llm):
        print(llm.__class__.__name__)
        classifier = ReviewClassifier()
        response = classifier("A real gem of Cantonese cooking. Food was really hot. You can taste the 'wok hei' in the food. Service was friendly and quick. Definitely coming again.")
        print("**Sentiment**: ", response.sentiment)
        print("**Topics**: ", response.topics)
        print("**Explanation**: ", response.explanation)


