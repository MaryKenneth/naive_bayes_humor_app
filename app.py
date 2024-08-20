#import flask module
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from data_preprocessing import tokens,lemmatize_spacy

# instance of flask application 
app = Flask(__name__)

# Load the model
with open('naive_bayes_humor_classifier_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Define label mappings
label_map_5 = {
    0: "Self-enhancing",
    1: "Self-deprecating",
    2: "Affiliative",
    3: "Aggressive",
    4: "Neutral"
}

# Label Explanation Mapping 
explanation_map = {
    0: "This style involves making jokes that make yourself feel good or help you cope with difficult situations. It's often lighthearted and positive, focusing on finding humor in life without putting others down.",
    1: "This style involves making jokes at your own expense. While it can be funny, it often highlights your own flaws or shortcomings.",
    2: "This style is all about sharing jokes that bring people together and make everyone laugh. It's friendly, inclusive, and helps build connections with others.",
    3: "This style includes jokes that might hurt or put others down, even if it's meant to be funny. It can come across as mean or sarcastic.",
    4: "This style is just general humor that doesn't strongly fall into any of the other categories. It's neither overly positive nor negative, and it's usually inoffensive and mild."
}

# Label Encouragement
encouragement_map = {
    0: "Great job! This kind of humor helps you stay positive and enjoy life more.",
    1: "It's okay to laugh at yourself sometimes, but be kind to yourself. Too much self-deprecating humor can hurt your self-esteem.",
    2: "Awesome! This kind of humor brings people closer and spreads positivity.",
    3: "Try to avoid humor that could hurt others. Positive humor makes everyone feel good.",
    4: "Nice! Neutral humor is safe and can be enjoyed by most people."
}


def predict_naive_bayes_single(example, log_probs, prob_words):
    # Tokenize the example
    example_tokens = tokens(example)

    # Calculate the log likelihoods for each class
    class_likelihoods = {}
    for class_label, log_prob in log_probs.items():
        class_likelihood = log_prob + sum(prob_words[class_label].get(word, 0) for word in example_tokens)
        class_likelihoods[class_label] = class_likelihood

    # Make a prediction based on the class with the highest likelihood
    prediction = max(class_likelihoods, key=class_likelihoods.get)
    
    # Calculate the confidence score (probability) of the predicted class
    confidence_score = class_likelihoods[prediction]

    return prediction, confidence_score

# home route that returns below text when root url is accessed
@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/classify_joke", methods=["POST"])
def classify_joke():
    data = request.json
    joke = data['joke']

    # Preprocess the joke
    preprocessed_joke = lemmatize_spacy(joke)
    preprocessed_joke = ", ".join(preprocessed_joke)

    # Make predictions
    prediction_5, probability = predict_naive_bayes_single(preprocessed_joke, model_data['log_probs_5'], model_data['prob_words_5'])

    # Convert numpy types to Python native types
    prediction_5 = int(prediction_5) if isinstance(prediction_5, (np.int32, np.int64)) else prediction_5

    # Map numeric predictions to string labels
    label_5 = label_map_5.get(prediction_5, "Unknown")
    explanation = explanation_map.get(prediction_5, "Unknown")
    encouragement = encouragement_map.get(prediction_5, "Unknown")

    return jsonify({
        'style': f"{label_5}",
        'explain': f"{explanation}",
        'encourage': f"{encouragement}",
        'probability': f"Model Confidence: {probability:.2f}"
    })


if __name__ == "__main__":
    app.run(debug=True)