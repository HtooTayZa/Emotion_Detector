"""Unit tests for EmotionDetection.emotion_detector.

External HTTP calls are mocked so tests run without network access
and remain deterministic — this is best practice for unit testing.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EmotionDetection import emotion_detector  


def _mock_response(status_code, emotions):
    """Build a mock requests.Response with the Watson NLP JSON shape."""
    mock = MagicMock()
    mock.status_code = status_code
    if status_code == 400:
        mock.text = '{"error":"bad request"}'
    else:
        import json
        mock.text = json.dumps({
            "emotionPredictions": [{"emotion": emotions}]
        })
    return mock


class TestEmotionDetector(unittest.TestCase):
    """Verify dominant_emotion is correct for each target statement."""

    def _run(self, text, emotions):
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=_mock_response(200, emotions)) as _:
            return emotion_detector(text)

    def test_joy_statement(self):
        result = self._run(
            "I am glad this happened",
            {"anger": 0.05, "disgust": 0.03, "fear": 0.02, "joy": 0.85, "sadness": 0.05},
        )
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_anger_statement(self):
        result = self._run(
            "I am really mad about this",
            {"anger": 0.90, "disgust": 0.03, "fear": 0.02, "joy": 0.01, "sadness": 0.04},
        )
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust_statement(self):
        result = self._run(
            "I feel disgusted just hearing about this",
            {"anger": 0.10, "disgust": 0.75, "fear": 0.05, "joy": 0.02, "sadness": 0.08},
        )
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_sadness_statement(self):
        result = self._run(
            "I am so sad about this",
            {"anger": 0.05, "disgust": 0.03, "fear": 0.07, "joy": 0.02, "sadness": 0.83},
        )
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_fear_statement(self):
        result = self._run(
            "I am really afraid that this will happen",
            {"anger": 0.04, "disgust": 0.02, "fear": 0.88, "joy": 0.01, "sadness": 0.05},
        )
        self.assertEqual(result["dominant_emotion"], "fear")

    def test_blank_input_returns_none(self):
        """Blank / empty text triggers a 400 — all fields should be None."""
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=_mock_response(400, {})):
            result = emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])
        self.assertIsNone(result["anger"])


if __name__ == "__main__":
    unittest.main()
