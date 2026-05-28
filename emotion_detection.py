"""
Emotion Detection module using the Watson NLP Embedded AI library.
"""

import json
import requests


def emotion_detector(text_to_analyse):
    """Send text to the Watson NLP emotion endpoint and return results.

    Args:
        text_to_analyse (str): The text to be analysed for emotion.

    Returns:
        dict: Keys anger, disgust, fear, joy, sadness (float scores each)
              and dominant_emotion (str). All values are None when the
              input is blank or the service returns a 400 status.
    """
    url = (
        "https://sn-watson-emotion.labs.skills.network"
        "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = {"raw_document": {"text": text_to_analyse}}

    response = requests.post(url, json=payload, headers=headers, timeout=30)

    # --- Task 7: Error handling ---
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response_json = json.loads(response.text)

    # Navigate to the emotion scores inside the Watson NLP response
    emotions = (
        response_json
        .get("emotionPredictions", [{}])[0]
        .get("emotion", {})
    )

    anger = emotions.get("anger", 0.0)
    disgust = emotions.get("disgust", 0.0)
    fear = emotions.get("fear", 0.0)
    joy = emotions.get("joy", 0.0)
    sadness = emotions.get("sadness", 0.0)

    # Task 3: Determine the dominant emotion
    scores = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
    }
    dominant_emotion = max(scores, key=scores.get)

    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion,
    }
