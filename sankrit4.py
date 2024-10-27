import wikipediaapi
import requests
from bs4 import BeautifulSoup
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a Wikipedia API object with a user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='sa',  # Set the language code for Sanskrit
    user_agent='YourAppName/1.0'
)

# Specify the title of the seed page you want to access (a Sanskrit page)
seed_page_title = "Om"

# Retrieve the seed page object
seed_page = wiki_wiki.page(seed_page_title)

# Access the links on the seed page
links = seed_page.links

# Create a list to store URLs and their corresponding cosine similarity scores
url_similarity_list = []

# Define a function to calculate cosine similarity between two text vectors
def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cosine_score[0][0]

# Calculate cosine similarity scores and prioritize URLs based on similarity
for link_title in links:
    link_url = "https://sa.wikipedia.org/wiki/" + link_title
    
    # Retrieve the page object for the linked page
    linked_page = wiki_wiki.page(link_title)
    
    # Extract the content of the linked page
    linked_page_content = linked_page.text
    
    # Calculate cosine similarity between the "Om" page and the linked page
    similarity_score = calculate_cosine_similarity(seed_page.text, linked_page_content)
    
    # Add the URL and cosine similarity score to the list
    url_similarity_list.append((link_url, similarity_score))

# Sort the list of URLs based on cosine similarity scores (higher score first)
url_similarity_list.sort(key=lambda x: x[1], reverse=True)

# Retrieve and print the top 5 URLs along with cosine similarity scores
top_5_urls = url_similarity_list[:5]
print("\nTop 5 Page URLs Based on Cosine Similarity:")
for url, similarity_score in top_5_urls:
    print(f"URL: {url}, Cosine Similarity Score: {similarity_score}")
