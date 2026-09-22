import json
import unittest
from unittest.mock import patch

from EmotionDetection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    test_cases = {
        "I am glad this happened": "joy",
        "I am really mad about this": "anger",
        "I feel disgusted just hearing about this": "disgust",
        "I am so sad about this": "sadness",
        "I am really afraid that this will happen": "fear"
    }

    emotion_scores = {
        "joy": {
            "anger": 0.02,
            "disgust": 0.01,
            "fear": 0.01,
            "joy": 0.95,
            "sadness": 0.01
        },
        "anger": {
            "anger": 0.95,
            "disgust": 0.01,
            "fear": 0.02,
            "joy": 0.01,
            "sadness": 0.01
        },
        "disgust": {
            "anger": 0.01,
            "disgust": 0.95,
            "fear": 0.01,
            "joy": 0.01,
            "sadness": 0.02
        },
        "sadness": {
            "anger": 0.01,
            "disgust": 0.01,
            "fear": 0.02,
            "joy": 0.01,
            "sadness": 0.95
        },
        "fear": {
            "anger": 0.01,
            "disgust": 0.01,
            "fear": 0.95,
            "joy": 0.01,
            "sadness": 0.02
        }
    }

    def mock_response(self, text):
        dominant_emotion = self.test_cases[text]
        response = unittest.mock.Mock()
        response.text = json.dumps({
            "emotionPredictions": [
                {
                    "emotion": self.emotion_scores[dominant_emotion]
                }
            ]
        })
        return response

    def assert_dominant_emotion(self, text, expected_emotion):
        with patch("EmotionDetection.emotion_detection.requests.post") as mock_post:
            mock_post.side_effect = lambda url, json=None, headers=None: self.mock_response(
                json["raw_document"]["text"]
            )
            result = emotion_detector(text)
            self.assertEqual(result["dominant_emotion"], expected_emotion)

    def test_joy(self):
        self.assert_dominant_emotion("I am glad this happened", "joy")

    def test_anger(self):
        self.assert_dominant_emotion("I am really mad about this", "anger")

    def test_disgust(self):
        self.assert_dominant_emotion(
            "I feel disgusted just hearing about this",
            "disgust"
        )

    def test_sadness(self):
        self.assert_dominant_emotion("I am so sad about this", "sadness")

    def test_fear(self):
        self.assert_dominant_emotion(
            "I am really afraid that this will happen",
            "fear"
        )


if __name__ == "__main__":
    unittest.main()
