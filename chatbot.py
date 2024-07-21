import re
from collections import Counter

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from faq_data import faq_data


# Scrape multiple websites and print the content in the terminal
def scrape_multiple_websites(urls):
    all_content = {}
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        content = {}

        for header in soup.find_all(["h1", "h2", "h3", "p"]):
            if header.name.startswith("h"):  # Process headers
                title = header.get_text().strip()
                paragraph = (
                    header.find_next("p").get_text().strip()
                    if header.find_next("p")
                    else ""
                )
                # Store content with URL
                content[title] = {"text": paragraph, "url": url}
            elif (
                header.name == "p" and header.get_text().strip() not in content
            ):  # Process paragraphs directly
                title = header.get_text().strip()
                # Store content with URL
                content[title] = {"text": header.get_text().strip(), "url": url}

        all_content.update(content)
    return all_content


# URLs to scrape
urls_to_scrape = [
    "https://webwizard.ie/",
    "https://webwizard.ie/custom-web-apps-development/",
    "https://webwizard.ie/business-website-design-development/",
    "https://webwizard.ie/shopify-e-commerce-website-development-ireland/",
    "https://webwizard.ie/personal-website-design-development/",
    "https://webwizard.ie/graphics-design/",
    "https://webwizard.ie/about-us/",
    "https://webwizard.ie/contact-us/",
]

# Combined dataset for FAQs and responses
scraped_content = scrape_multiple_websites(urls_to_scrape)


# Process the scraped content
for title, content in scraped_content.items():
    faq_data.setdefault(
        title, {"question": title, "response": content["text"], "url": content["url"]}
    )

# Preprocess text data
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    tokens = word_tokenize(text)
    filtered_tokens = [
        token for token in tokens if token.isalnum() and token not in stop_words
    ]
    return " ".join(filtered_tokens)

# Prepare training data
X_train = [preprocess_text(value["question"]) for value in faq_data.values()]
y_train = list(faq_data.keys())

# Initialize and train the classifier
text_clf = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
text_clf.fit(X_train, y_train)

# Evaluate the classifier
# Simplified accuracy calculation
accuracy = text_clf.score(X_train, y_train)


# We are using a pre-trained BERT model to generate embeddings for sentences,
# which can then be used to calculate similarity between user input and FAQ questions.

# Load pre-trained BERT model for embedding sentences
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Create embeddings for FAQ responses
faq_questions = [value["question"] for value in faq_data.values()]
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)


def chatbot(user_input):
    user_input_processed = preprocess_text(user_input)
    user_input_embedding = model.encode(user_input_processed, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(user_input_embedding, faq_embeddings)[0]
    max_index = cosine_scores.argmax()
    max_score = cosine_scores[max_index]
    # Debugging print statements
    # print(f"Max score: {max_score}")
    # print(f"Max index: {max_index}")

    confidence_threshold = 0.4

    if max_score >= confidence_threshold:
        # Retrieve FAQ data from faq_data
        faq_entry = list(faq_data.values())[max_index]
        response = faq_entry["response"]

        # remove unwanted whitespaces and line breaks
        response = ' '.join(response.split())

        # Check if the response is from a scraped website and include the URL if available
        if "url" in faq_entry:
            response += f"\n\n[Source: {faq_entry['url']}]"
        return response
    else:
        return "I'm sorry, I didn't understand your question. Could you please rephrase it?"

all_text = " ".join(
    [preprocess_text(content["text"]) for content in scraped_content.values()]
)

# Count word frequencies
word_counts = Counter(all_text.split())
# Use 10 most common words for plotting
most_common_words = word_counts.most_common(10)

# Separate words and counts for plotting
words, counts = zip(*most_common_words)

# Generate bar chart
plt.figure(figsize=(10, 8))
plt.bar(words, counts)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Words")
plt.xticks(rotation=45)
plt.show()

# Update the main loop to print the bot's response
if __name__ == "__main__":
    print("Welcome to the FAQ Chatbot. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        bot_response = chatbot(user_input)
        print("Bot:", bot_response)
