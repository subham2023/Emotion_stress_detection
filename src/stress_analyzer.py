"""
Stress Analyzer Module for Emotion Recognition System.

This module calculates stress levels based on emotion probabilities,
facial features, and temporal analysis.
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime

import numpy as np

from config import (
    EMOTIONS,
    EMOTION_STRESS_WEIGHTS,
    STRESS_LEVELS,
    get_stress_level,
    LOG_FILE,
    LOG_LEVEL,
    LOG_FORMAT,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class StressAnalysisError(Exception):
    """Base exception for stress analysis errors."""

    pass


# ============================================================================
# STRESS ANALYZER
# ============================================================================


class StressAnalyzer:
    """
    Analyze stress levels from emotion predictions and facial features.

    The stress calculation is based on:
    1. Emotion probabilities and their stress weights
    2. Facial landmarks (eyebrow distance, jaw tension)
    3. Temporal patterns (stress over time)
    """

    def __init__(self, history_size: int = 30):
        """
        Initialize stress analyzer.

        Args:
            history_size: Number of frames to keep in history for temporal analysis
        """
        self.history_size = history_size
        self.stress_history: deque = deque(maxlen=history_size)
        self.emotion_history: deque = deque(maxlen=history_size)
        self.timestamp_history: deque = deque(maxlen=history_size)

    @staticmethod
    def calculate_emotion_stress(emotion_probs: np.ndarray) -> float:
        """
        Calculate stress score based on emotion probabilities.

        Args:
            emotion_probs: Array of emotion probabilities (7 emotions)

        Returns:
            Stress score (0-100)

        Raises:
            StressAnalysisError: If input is invalid
        """
        try:
            if len(emotion_probs) != len(EMOTIONS):
                raise StressAnalysisError(
                    f"Expected {len(EMOTIONS)} emotion probabilities, got {len(emotion_probs)}"
                )

            if not np.isclose(emotion_probs.sum(), 1.0, atol=0.01):
                raise StressAnalysisError("Emotion probabilities must sum to 1.0")

            # Calculate weighted stress score
            stress_score = 0.0
            for emotion_idx, emotion_name in enumerate(EMOTIONS):
                weight = EMOTION_STRESS_WEIGHTS.get(emotion_name, 0.0)
                prob = emotion_probs[emotion_idx]
                stress_score += weight * prob

            # Normalize to 0-100 range
            # Weights range from -0.8 (happy) to 0.95 (fear)
            # So raw score ranges from -0.8 to 0.95
            min_score = -0.8
            max_score = 0.95
            normalized_score = (
                (stress_score - min_score) / (max_score - min_score)
            ) * 100
            normalized_score = np.clip(normalized_score, 0, 100)

            return float(normalized_score)

        except Exception as e:
            raise StressAnalysisError(f"Failed to calculate emotion stress: {str(e)}")

    @staticmethod
    def calculate_facial_tension(landmarks: Optional[Dict[str, float]]) -> float:
        """
        Calculate facial tension from landmarks.

        Args:
            landmarks: Dictionary with facial landmark measurements
                      (eyebrow_distance, jaw_tension, etc.)

        Returns:
            Facial tension score (0-100)
        """
        if landmarks is None or not landmarks:
            return 0.0

        tension_score = 0.0

        # Eyebrow distance (lower = more tension)
        if "eyebrow_distance" in landmarks:
            eyebrow_dist = landmarks["eyebrow_distance"]
            # Normalize: assume normal range is 0.1-0.3
            tension_score += max(0, (0.3 - eyebrow_dist) / 0.2 * 50)

        # Jaw tension (higher = more tension)
        if "jaw_tension" in landmarks:
            jaw_tension = landmarks["jaw_tension"]
            tension_score += min(100, jaw_tension * 50)

        # Mouth opening (lower = more tension)
        if "mouth_opening" in landmarks:
            mouth_opening = landmarks["mouth_opening"]
            # Normalize: assume normal range is 0.05-0.15
            tension_score += max(0, (0.15 - mouth_opening) / 0.1 * 30)

        return min(100, tension_score / 3)  # Average and cap at 100

    @staticmethod
    def calculate_combined_stress(
        emotion_stress: float,
        facial_tension: float = 0.0,
        emotion_weight: float = 0.7,
        tension_weight: float = 0.3,
    ) -> float:
        """
        Calculate combined stress score from emotion and facial tension.

        Args:
            emotion_stress: Emotion-based stress score (0-100)
            facial_tension: Facial tension score (0-100)
            emotion_weight: Weight for emotion stress
            tension_weight: Weight for facial tension

        Returns:
            Combined stress score (0-100)
        """
        combined_score = (
            emotion_stress * emotion_weight + facial_tension * tension_weight
        )
        return float(np.clip(combined_score, 0, 100))

    def analyze_frame(
        self,
        emotion_probs: np.ndarray,
        landmarks: Optional[Dict[str, float]] = None,
    ) -> Dict[str, any]:
        """
        Analyze a single frame and return stress metrics.

        Args:
            emotion_probs: Array of emotion probabilities
            landmarks: Facial landmarks (optional)

        Returns:
            Dictionary with stress analysis results

        Raises:
            StressAnalysisError: If analysis fails
        """
        try:
            # Calculate emotion stress
            emotion_stress = self.calculate_emotion_stress(emotion_probs)

            # Calculate facial tension
            facial_tension = self.calculate_facial_tension(landmarks)

            # Calculate combined stress
            combined_stress = self.calculate_combined_stress(
                emotion_stress, facial_tension
            )

            # Get dominant emotion
            dominant_emotion_idx = np.argmax(emotion_probs)
            dominant_emotion = EMOTIONS[dominant_emotion_idx]
            dominant_emotion_prob = float(emotion_probs[dominant_emotion_idx])

            # Get stress level
            stress_level = get_stress_level(combined_stress)

            # Add to history
            self.stress_history.append(combined_stress)
            self.emotion_history.append(dominant_emotion)
            self.timestamp_history.append(datetime.now())

            result = {
                "timestamp": datetime.now().isoformat(),
                "emotion_stress": emotion_stress,
                "facial_tension": facial_tension,
                "combined_stress": combined_stress,
                "stress_level": stress_level,
                "dominant_emotion": dominant_emotion,
                "dominant_emotion_prob": dominant_emotion_prob,
                "emotion_probabilities": {
                    EMOTIONS[i]: float(emotion_probs[i]) for i in range(len(EMOTIONS))
                },
            }

            return result

        except Exception as e:
            raise StressAnalysisError(f"Frame analysis failed: {str(e)}")

    def get_temporal_analysis(self) -> Dict[str, any]:
        """
        Analyze stress patterns over time.

        Returns:
            Dictionary with temporal analysis results
        """
        if not self.stress_history:
            return {
                "mean_stress": 0.0,
                "max_stress": 0.0,
                "min_stress": 0.0,
                "std_stress": 0.0,
                "trend": "stable",
                "dominant_emotion": "neutral",
            }

        stress_array = np.array(list(self.stress_history))
        emotion_array = np.array(list(self.emotion_history))

        mean_stress = float(np.mean(stress_array))
        max_stress = float(np.max(stress_array))
        min_stress = float(np.min(stress_array))
        std_stress = float(np.std(stress_array))

        # Detect trend
        if len(stress_array) >= 2:
            recent_stress = np.mean(stress_array[-5:])
            older_stress = (
                np.mean(stress_array[:-5]) if len(stress_array) > 5 else stress_array[0]
            )
            trend_diff = recent_stress - older_stress

            if trend_diff > 5:
                trend = "increasing"
            elif trend_diff < -5:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Get most common emotion
        unique, counts = np.unique(emotion_array, return_counts=True)
        dominant_emotion = unique[np.argmax(counts)]

        return {
            "mean_stress": mean_stress,
            "max_stress": max_stress,
            "min_stress": min_stress,
            "std_stress": std_stress,
            "trend": trend,
            "dominant_emotion": str(dominant_emotion),
            "history_size": len(self.stress_history),
        }

    def get_recommendations(self, stress_level: str) -> List[str]:
        """
        Get recommendations based on stress level.

        Args:
            stress_level: Current stress level

        Returns:
            List of recommendations
        """
        recommendations = {
            "low": [
                "You seem relaxed. Keep up the good work!",
                "Your stress levels are healthy. Maintain this state.",
                "Great job managing your stress!",
            ],
            "moderate": [
                "Consider taking a short break.",
                "Try some deep breathing exercises.",
                "Take a walk or stretch to relieve tension.",
                "Listen to some calming music.",
            ],
            "high": [
                "Your stress levels are elevated. Take a break immediately.",
                "Try progressive muscle relaxation.",
                "Consider stepping away from your current task.",
                "Practice deep breathing: inhale for 4, hold for 4, exhale for 4.",
                "Reach out to someone for support.",
            ],
            "critical": [
                "Your stress levels are critical. Stop what you're doing and take a break.",
                "Seek immediate support or contact a mental health professional.",
                "Try emergency stress relief: splash cold water on your face, or go outside.",
                "Consider calling a trusted friend or family member.",
                "If you're in crisis, please reach out to a mental health hotline.",
            ],
        }

        return recommendations.get(stress_level, [])

    def reset_history(self) -> None:
        """Reset all history buffers."""
        self.stress_history.clear()
        self.emotion_history.clear()
        self.timestamp_history.clear()
        logger.info("Stress analyzer history reset")


# ============================================================================
# STRESS REPORT GENERATOR
# ============================================================================


class StressReportGenerator:
    """
    Generate detailed stress analysis reports.
    """

    @staticmethod
    def generate_session_report(
        stress_history: List[float],
        emotion_history: List[str],
        session_duration: float,
    ) -> Dict[str, any]:
        """
        Generate a session report.

        Args:
            stress_history: List of stress scores over time
            emotion_history: List of detected emotions
            session_duration: Duration of session in seconds

        Returns:
            Session report dictionary
        """
        if not stress_history:
            return {"error": "No data available for report"}

        stress_array = np.array(stress_history)
        emotion_array = np.array(emotion_history)

        # Calculate statistics
        stats = {
            "mean_stress": float(np.mean(stress_array)),
            "max_stress": float(np.max(stress_array)),
            "min_stress": float(np.min(stress_array)),
            "std_stress": float(np.std(stress_array)),
            "median_stress": float(np.median(stress_array)),
        }

        # Emotion distribution
        unique, counts = np.unique(emotion_array, return_counts=True)
        emotion_distribution = {
            emotion: int(count) for emotion, count in zip(unique, counts)
        }

        # Stress level distribution
        stress_level_dist = {
            "low": sum(1 for s in stress_array if s < 25),
            "moderate": sum(1 for s in stress_array if 25 <= s < 50),
            "high": sum(1 for s in stress_array if 50 <= s < 75),
            "critical": sum(1 for s in stress_array if s >= 75),
        }

        report = {
            "session_duration_seconds": session_duration,
            "total_frames": len(stress_history),
            "statistics": stats,
            "emotion_distribution": emotion_distribution,
            "stress_level_distribution": stress_level_dist,
            "overall_stress_level": get_stress_level(stats["mean_stress"]),
        }

        return report

    @staticmethod
    def format_report_text(report: Dict[str, any]) -> str:
        """
        Format report as text.

        Args:
            report: Report dictionary

        Returns:
            Formatted text report
        """
        if "error" in report:
            return f"Error: {report['error']}"

        text = []
        text.append("=" * 80)
        text.append("STRESS ANALYSIS REPORT")
        text.append("=" * 80)

        # Session info
        text.append(
            f"\nSession Duration: {report['session_duration_seconds']:.1f} seconds"
        )
        text.append(f"Total Frames: {report['total_frames']}")

        # Statistics
        text.append("\nSTRESS STATISTICS:")
        stats = report["statistics"]
        text.append(f"  Mean Stress: {stats['mean_stress']:.2f}")
        text.append(f"  Max Stress: {stats['max_stress']:.2f}")
        text.append(f"  Min Stress: {stats['min_stress']:.2f}")
        text.append(f"  Std Dev: {stats['std_stress']:.2f}")
        text.append(f"  Median: {stats['median_stress']:.2f}")

        # Emotion distribution
        text.append("\nEMOTION DISTRIBUTION:")
        for emotion, count in report["emotion_distribution"].items():
            pct = (count / report["total_frames"]) * 100
            text.append(f"  {emotion}: {count} ({pct:.1f}%)")

        # Stress level distribution
        text.append("\nSTRESS LEVEL DISTRIBUTION:")
        for level, count in report["stress_level_distribution"].items():
            pct = (count / report["total_frames"]) * 100
            text.append(f"  {level}: {count} ({pct:.1f}%)")

        # Overall assessment
        text.append(f"\nOVERALL STRESS LEVEL: {report['overall_stress_level'].upper()}")

        text.append("\n" + "=" * 80)

        return "\n".join(text)
