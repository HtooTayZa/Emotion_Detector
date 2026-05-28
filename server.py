"""Executing this module starts the Emotion Analysis Flask application
and deploys it on localhost:5000.
"""

from flask import Flask, render_template, request
from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/emotionDetector")
def emotion_detector_route():
    """Receive text from the HTML interface, run emotion detection,
    and return a formatted plain-text result to the caller.
    """
    text_to_analyse = request.args.get("textToAnalyze", "")

    result = emotion_detector(text_to_analyse)

    # Task 7: handle blank / invalid input flagged by None values
    if result["dominant_emotion"] is None:
        return "Invalid text! Please try again."

    # Task 3: formatted output string
    return (
        f"For the given statement, the system response is "
        f"'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, "
        f"'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is <b>{result['dominant_emotion']}</b>."
    )


@app.route("/")
def render_index_page():
    """Render the main application page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
