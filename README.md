# Exp 7 Stock Market Prediction using Linear Regression and Real-Time Sentiment Analysis of Tweets

**Date:**25/08/2026

## AIM:

To implement **Stock Market Prediction using Linear Regression** to predict future stock prices using machine learning regression techniques and **Real-Time Sentiment Analysis of Tweets** to analyze user-provided text data such as tweets or reviews and classify their sentiment.

## DESIGN STEPS:

### Step 1:

Clone the repository from GitHub.

### Step 2:

Create a Python project in the preferred IDE (VS Code/PyCharm/Jupyter Notebook).

### Step 3:

Create the Python program for implementing Stock Market Prediction using Linear Regression and Real-Time Sentiment Analysis using suitable Python libraries.

### Step 4:

Load the historical stock market dataset and select the required features such as **Open, High, Low, Volume**, and **Close** price.

### Step 5:

Preprocess the stock market dataset and split it into training and testing data.

### Step 6:

Train the **Linear Regression** model using the training data and predict the stock prices for the testing data.

### Step 7:

Evaluate the stock prediction model using suitable regression metrics and visualize the actual and predicted stock prices.

### Step 8:

Accept user-provided text data such as **tweets, reviews, or comments** as input for sentiment analysis.

### Step 9:

Preprocess the input text by converting it into lowercase and removing unnecessary characters and punctuation.

### Step 10:

Calculate the **sentiment polarity score** of the input text using Natural Language Processing techniques.

### Step 11:

Classify the input text as **Positive, Negative, or Neutral** based on the polarity score.

### Step 12:

Execute the program and analyze the stock price prediction and sentiment analysis results.


###DEVELOPED BY:ANGELIN GRACY.R
###REGISTER NUMBER:212225240009

## PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read stock data
data = pd.read_csv("stock_data_big.csv")

# Display first 5 rows
print(data.head())

# Input features
X = data[["Open", "High", "Low", "Volume"]]

# Target variable
y = data["Close"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict stock prices
predictions = model.predict(X_test)

# Calculate Mean Squared Error
print("\nMean Squared Error:", mean_squared_error(y_test, predictions))

# Calculate R2 Score
print("R2 Score:", r2_score(y_test, predictions))

# Plot Actual vs Predicted prices
plt.figure(figsize=(8, 5))

plt.plot(
    y_test.values[:50],
    label="Actual Prices",
    color="blue"
)

plt.plot(
    predictions[:50],
    label="Predicted Prices",
    color="red"
)

plt.title("Stock Price Prediction (Actual vs Predicted)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

plt.show()
```


<img width="515" height="177" alt="image" src="https://github.com/user-attachments/assets/8f45fc38-01ab-46e6-8ce2-7298cdb09ae1" />


<img width="695" height="468" alt="download" src="https://github.com/user-attachments/assets/a876cf53-867d-45d3-9bce-70d52bdca8d8" />


```
import pandas as pd
import matplotlib.pyplot as plt
import re

# Read tweets data
data = pd.read_csv("tweets_big.csv")

# Positive and negative words
positive_words = {
    "good", "great", "excellent", "awesome", "amazing",
    "love", "lovely", "best", "happy", "fast", "smooth",
    "nice", "wonderful", "perfect", "fantastic", "positive",
    "success", "successful", "easy", "helpful", "thank"
}

negative_words = {
    "bad", "poor", "worst", "hate", "slow", "rude",
    "terrible", "awful", "horrible", "sad", "negative",
    "problem", "problems", "fail", "failed", "failure",
    "difficult", "hard", "angry", "delay", "delayed",
    "broken", "disappointing", "disappointed"
}

# Sentiment function
def get_sentiment(text):

    text = str(text).lower()

    # Remove punctuation
    words = re.findall(r'\b[a-z]+\b', text)

    positive_count = 0
    negative_count = 0

    for word in words:
        if word in positive_words:
            positive_count += 1

        if word in negative_words:
            negative_count += 1

    if positive_count > negative_count:
        return "Positive"

    elif negative_count > positive_count:
        return "Negative"

    else:
        return "Neutral"


# Apply sentiment analysis
data["Sentiment"] = data["text"].apply(get_sentiment)

# Count sentiments
sentiment_counts = data["Sentiment"].value_counts()

print("Sentiment")
print(sentiment_counts)

# Make sure all three categories appear
categories = ["Negative", "Positive", "Neutral"]

counts = [
    sentiment_counts.get("Negative", 0),
    sentiment_counts.get("Positive", 0),
    sentiment_counts.get("Neutral", 0)
]

# Plot graph
plt.figure(figsize=(6, 4))

plt.bar(
    categories,
    counts,
    color=["green", "red", "gray"]
)

plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment Type")
plt.ylabel("Number of Tweets/Reviews")

plt.show()

# Display sample results
print("\nSample Results:")
print(data[["text", "Sentiment"]].head())
```

<img width="592" height="641" alt="image" src="https://github.com/user-attachments/assets/0f252bf4-9445-45a1-8a98-348faa0f5646" />

## RESULT:

The **Stock Market Prediction using Linear Regression** and **Real-Time Sentiment Analysis of Tweets** were implemented successfully. The Linear Regression model was used to predict stock prices using historical stock market data, while the sentiment analysis system successfully analyzed user-provided tweets or reviews and classified them as **Positive, Negative, or Neutral**.
