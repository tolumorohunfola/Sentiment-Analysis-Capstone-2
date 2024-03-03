# A program to do sentiment analysis on the amazon product reviews.

"""
1. Implement a sentiment analysis model using spacy

2. Preprocess the text data

3. Create a function for sentiment analysis that takes a product review
as input and predicts its sentiment

4. Test your model on sample product reviews to verify its accuracy in
predicting sentiment

5. Write a brief report or summary in a pdf file which will include:

5.1 A description of the dataset used
5.2 Details of the preprocessing steps
5.3 Evaluation of results
5.4 Insights into the model's strengths and limitations

"""
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# loaded the sm spacy model
nlp = spacy.load("en_core_web_sm")


# loading the data
dataframe = pd.read_csv("amazon_product_reviews.csv", delimiter=",")

# * preprocessing the data
# narrowing the dataframe to only the information we need and removing missing
# values from the required column

# text cleaning by creating a dataset containing only the dat necessary to
# answer the question
reviews_data = dataframe["reviews.text"]

# text cleaning by removing all missing values
filled_reviews_data = reviews_data.dropna()


# * removing stop words and punctuation, lower casing the data and removing
# * extra space

# creating a list of stop words
stopwords = nlp.Defaults.stop_words
# new data frame for the cleaned reviews
clean_reviews = []


for review in filled_reviews_data:
    # creating a new review sentence
    # list comprehension to add each word in the review to the list if it
    # is not in the list of stopwords and join the list of words back into a
    # string
    new_review_words = [review_word for review_word in review.split() if review_word.lower().strip() not in stopwords]
    new_review = " ".join(new_review_words)

    # joins all the words in the new review words list to make a new version of the reviews
    # new_review = " ".join(new_review_words)
    # * removing punctuation
    # list comprehension to add each word / token in the review to the list
    # if it is not punctuation and join the words back into one text
    new_review_with_no_punct = [token for token in nlp(new_review) if not token.is_punct]
    new_review_with_no_punct = " ".join(token.text for token in new_review_with_no_punct)

    # adds the lowered and stripped version of the review not including
    # stopwords
    clean_reviews.append(new_review_with_no_punct.strip().lower())


# function to assess polarity and subjectivity
def sentiment(review_number):

    # assigning the review variable to the review input
    review = clean_reviews[review_number]
    # creates object from the review
    review_nlp = nlp(review)
    # prints out the original and cleaned text in the review.
    print(f"Raw text: {filled_reviews_data.iloc[review_number]}")
    print(f"Clean text: {review_nlp.text}")
    print("***********************************************")

    # prints the subjectivity the sentiment of the review is on a scale of 0 and 1
    print(review_nlp._.blob.sentiment)
    # if statement to check polarity index and save the polarity type
    if float(review_nlp._.blob.polarity) < 0:
        polarity_type = "negative"
    elif float(review_nlp._.blob.polarity) > 0:
        polarity_type = "positive"
    else:
        polarity_type = "neutral"
        extent_of_polarity = ""
    # if statement to check polarity index and save the extent of the polarity
    absolute_polarity = abs(float(review_nlp._.blob.polarity))
    if absolute_polarity < 0.4 and absolute_polarity != 0:
        extent_of_polarity = "slightly"
    elif absolute_polarity < 0.7:
        extent_of_polarity = "quite"
    elif absolute_polarity >= 0.7:
        extent_of_polarity = "very"
    # print the predicted sentiment of the review
    print(f"The sentiment is predicted to be {extent_of_polarity} {polarity_type}.")
    print("*********************************************** \n \n")

# adds the spacy text blob pipe to the nlp pipeline.
nlp.add_pipe('spacytextblob')

# testing of the sentiment function
sentiment(0)
sentiment(3)
sentiment(7)
sentiment(399)
sentiment(80)

