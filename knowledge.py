# godfrey_system/knowledge.py
"""
Godfrey System - Knowledge and Pattern Recognition
-------------------------------------------------
Knowledge representation and pattern recognition for the Godfrey system.
"""

import json
import logging
import time
import os
from typing import Dict, List, Any, Tuple, Optional, Set
import random
from datetime import datetime, timedelta

logger = logging.getLogger("Godfrey.Knowledge")

class KnowledgeBase:
    """Base class for knowledge storage and retrieval"""
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = time.time()
        self.last_updated = self.created_at
    
    def store(self, key: str, value: Any) -> bool:
        """Store a value in the knowledge base"""
        raise NotImplementedError("Subclasses must implement store()")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from the knowledge base"""
        raise NotImplementedError("Subclasses must implement retrieve()")
    
    def contains(self, key: str) -> bool:
        """Check if the knowledge base contains a key"""
        raise NotImplementedError("Subclasses must implement contains()")
    
    def remove(self, key: str) -> bool:
        """Remove a key from the knowledge base"""
        raise NotImplementedError("Subclasses must implement remove()")
    
    def clear(self):
        """Clear all data from the knowledge base"""
        raise NotImplementedError("Subclasses must implement clear()")

class FileKnowledgeBase(KnowledgeBase):
    """Knowledge base that stores data in a JSON file"""
    
    def __init__(self, name: str, file_path: str):
        super().__init__(name)
        self.file_path = file_path
        self.data = {}
        self._load()
    
    def _load(self):
        """Load data from the file"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self.data = json.load(f)
                logger.info(f"Loaded knowledge base from {self.file_path}")
            except Exception as e:
                logger.error(f"Error loading knowledge base from {self.file_path}: {e}")
                self.data = {}
    
    def _save(self):
        """Save data to the file"""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            self.last_updated = time.time()
            logger.debug(f"Saved knowledge base to {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge base to {self.file_path}: {e}")
            return False
    
    def store(self, key: str, value: Any) -> bool:
        """Store a value in the knowledge base"""
        self.data[key] = value
        return self._save()
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from the knowledge base"""
        return self.data.get(key)
    
    def contains(self, key: str) -> bool:
        """Check if the knowledge base contains a key"""
        return key in self.data
    
    def remove(self, key: str) -> bool:
        """Remove a key from the knowledge base"""
        if key in self.data:
            del self.data[key]
            return self._save()
        return False
    
    def clear(self):
        """Clear all data from the knowledge base"""
        self.data = {}
        return self._save()
    
    def get_all_keys(self) -> List[str]:
        """Get all keys in the knowledge base"""
        return list(self.data.keys())
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all data in the knowledge base"""
        return self.data.copy()

class MemoryKnowledgeBase(KnowledgeBase):
    """Knowledge base that stores data in memory"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.data = {}
    
    def store(self, key: str, value: Any) -> bool:
        """Store a value in the knowledge base"""
        self.data[key] = value
        self.last_updated = time.time()
        return True
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from the knowledge base"""
        return self.data.get(key)
    
    def contains(self, key: str) -> bool:
        """Check if the knowledge base contains a key"""
        return key in self.data
    
    def remove(self, key: str) -> bool:
        """Remove a key from the knowledge base"""
        if key in self.data:
            del self.data[key]
            self.last_updated = time.time()
            return True
        return False
    
    def clear(self):
        """Clear all data from the knowledge base"""
        self.data = {}
        self.last_updated = time.time()
        return True
    
    def get_all_keys(self) -> List[str]:
        """Get all keys in the knowledge base"""
        return list(self.data.keys())
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all data in the knowledge base"""
        return self.data.copy()

class Pattern:
    """A pattern that can be recognized in data"""
    
    def __init__(self, pattern_id: str, name: str, description: str = ""):
        self.id = pattern_id
        self.name = name
        self.description = description
        self.created_at = time.time()
        self.last_updated = self.created_at
        self.recognition_count = 0
        self.last_recognized = None
    
    def matches(self, data: Any) -> bool:
        """Check if the data matches this pattern"""
        raise NotImplementedError("Subclasses must implement matches()")
    
    def confidence(self, data: Any) -> float:
        """Calculate confidence that the data matches this pattern (0.0 to 1.0)"""
        raise NotImplementedError("Subclasses must implement confidence()")
    
    def record_recognition(self):
        """Record that this pattern was recognized"""
        self.recognition_count += 1
        self.last_recognized = time.time()

class SimplePattern(Pattern):
    """A simple pattern based on key-value pairs"""
    
    def __init__(self, pattern_id: str, name: str, description: str = ""):
        super().__init__(pattern_id, name, description)
        self.required_matches = {}  # key -> value
        self.optional_matches = {}  # key -> value
        self.threshold = 0.5  # Minimum confidence threshold
    
    def add_required_match(self, key: str, value: Any):
        """Add a required key-value match"""
        self.required_matches[key] = value
        self.last_updated = time.time()
    
    def add_optional_match(self, key: str, value: Any):
        """Add an optional key-value match"""
        self.optional_matches[key] = value
        self.last_updated = time.time()
    
    def set_threshold(self, threshold: float):
        """Set the confidence threshold"""
        self.threshold = max(0.0, min(1.0, threshold))
    
    def matches(self, data: Dict[str, Any]) -> bool:
        """Check if the data matches this pattern"""
        if not isinstance(data, dict):
            return False
        
        confidence_value = self.confidence(data)
        return confidence_value >= self.threshold
    
    def confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence that the data matches this pattern (0.0 to 1.0)"""
        if not isinstance(data, dict):
            return 0.0
        
        # Check required matches
        for key, value in self.required_matches.items():
            if key not in data or data[key] != value:
                return 0.0
        
        # If no required matches, base score on optional matches
        if not self.required_matches and not self.optional_matches:
            return 0.0
        
        # Calculate score based on optional matches
        if not self.optional_matches:
            return 1.0  # All required matches were met
        
        matches = 0
        for key, value in self.optional_matches.items():
            if key in data and data[key] == value:
                matches += 1
        
        return matches / len(self.optional_matches)

class SequencePattern(Pattern):
    """A pattern that recognizes sequences of events"""
    
    def __init__(self, pattern_id: str, name: str, description: str = ""):
        super().__init__(pattern_id, name, description)
        self.sequence = []  # List of event descriptors
        self.max_time_gap = None  # Maximum time between events (in seconds), None for no limit
        self.allow_interspersed_events = True  # Whether unrelated events can occur between sequence events
        self.threshold = 0.7  # Minimum confidence threshold
    
    def add_sequence_event(self, event_descriptor: Dict[str, Any]):
        """Add an event to the sequence"""
        self.sequence.append(event_descriptor)
        self.last_updated = time.time()
    
    def set_max_time_gap(self, seconds: Optional[float]):
        """Set the maximum time gap between events"""
        self.max_time_gap = seconds
    
    def set_allow_interspersed_events(self, allow: bool):
        """Set whether unrelated events can occur between sequence events"""
        self.allow_interspersed_events = allow
    
    def matches(self, event_history: List[Dict[str, Any]]) -> bool:
        """Check if the event history matches this sequence pattern"""
        confidence_value = self.confidence(event_history)
        return confidence_value >= self.threshold
    
    def confidence(self, event_history: List[Dict[str, Any]]) -> float:
        """Calculate confidence that the event history matches this pattern (0.0 to 1.0)"""
        if not event_history or not self.sequence:
            return 0.0
        
        # Sort events by timestamp if available
        sorted_history = sorted(
            event_history, 
            key=lambda e: e.get("timestamp", 0)
        )
        
        # Try to find the sequence in the history
        sequence_pos = 0
        last_match_time = None
        matches = 0
        
        for event in sorted_history:
            event_time = event.get("timestamp", 0)
            
            # Check time gap if applicable
            if self.max_time_gap is not None and last_match_time is not None:
                if event_time - last_match_time > self.max_time_gap:
                    # Reset if time gap is too large
                    sequence_pos = 0
                    matches = 0
                    last_match_time = None
            
            # Check if this event matches the next in sequence
            current_pattern = self.sequence[sequence_pos]
            if self._event_matches_descriptor(event, current_pattern):
                matches += 1
                last_match_time = event_time
                sequence_pos += 1
                
                # If we've matched the entire sequence
                if sequence_pos >= len(self.sequence):
                    return 1.0
            elif not self.allow_interspersed_events:
                # If interspersed events are not allowed, reset on mismatch
                sequence_pos = 0
                matches = 0
                last_match_time = None
        
        # Return partial match confidence
        return matches / len(self.sequence)
    
    def _event_matches_descriptor(self, event: Dict[str, Any], descriptor: Dict[str, Any]) -> bool:
        """Check if an event matches a descriptor"""
        for key, value in descriptor.items():
            if key not in event or event[key] != value:
                return False
        return True

class PatternRecognizer:
    """Recognizes patterns in data"""
    
    def __init__(self):
        self.patterns = {}  # id -> pattern
    
    def register_pattern(self, pattern: Pattern):
        """Register a pattern"""
        self.patterns[pattern.id] = pattern
        logger.info(f"Registered pattern: {pattern.name} ({pattern.id})")
    
    def unregister_pattern(self, pattern_id: str):
        """Unregister a pattern"""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            logger.info(f"Unregistered pattern: {pattern_id}")
    
    def recognize(self, data: Any) -> List[Tuple[Pattern, float]]:
        """Recognize patterns in data"""
        matches = []
        
        for pattern in self.patterns.values():
            try:
                confidence = pattern.confidence(data)
                if confidence > 0:
                    matches.append((pattern, confidence))
                    pattern.record_recognition()
            except Exception as e:
                logger.error(f"Error in pattern recognition for {pattern.name}: {e}")
        
        # Sort by confidence in descending order
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

class AnomalyDetector:
    """Detects anomalies in data"""
    
    def __init__(self):
        self.baseline_data = {}  # key -> [values]
        self.window_size = 100  # Number of values to keep for each key
        self.threshold = 2.0  # Number of standard deviations for anomaly
    
    def add_data_point(self, key: str, value: float):
        """Add a data point to the baseline"""
        if key not in self.baseline_data:
            self.baseline_data[key] = []
        
        self.baseline_data[key].append(value)
        
        # Keep only the most recent values
        if len(self.baseline_data[key]) > self.window_size:
            self.baseline_data[key] = self.baseline_data[key][-self.window_size:]
    
    def is_anomaly(self, key: str, value: float) -> Tuple[bool, float]:
        """Check if a value is an anomaly"""
        if key not in self.baseline_data or len(self.baseline_data[key]) < 5:
            return False, 0.0
        
        values = self.baseline_data[key]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return value != mean, 0.0
        
        z_score = abs(value - mean) / std_dev
        return z_score > self.threshold, z_score

class ThreatDatabase:
    """Stores information about known threats"""
    
    def __init__(self):
        self.threats = {}  # id -> threat_info
    
    def add_threat(self, threat_id: str, threat_info: Dict[str, Any]):
        """Add a threat to the database"""
        self.threats[threat_id] = threat_info
        logger.info(f"Added threat to database: {threat_id}")
    
    def get_threat(self, threat_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a threat"""
        return self.threats.get(threat_id)
    
    def update_threat(self, threat_id: str, updates: Dict[str, Any]) -> bool:
        """Update information about a threat"""
        if threat_id in self.threats:
            self.threats[threat_id].update(updates)
            logger.info(f"Updated threat in database: {threat_id}")
            return True
        return False
    
    def remove_threat(self, threat_id: str) -> bool:
        """Remove a threat from the database"""
        if threat_id in self.threats:
            del self.threats[threat_id]
            logger.info(f"Removed threat from database: {threat_id}")
            return True
        return False
    
    def get_all_threats(self) -> Dict[str, Dict[str, Any]]:
        """Get all threats in the database"""
        return self.threats.copy()
    
    def search_threats(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for threats matching a query"""
        results = []
        
        for threat_id, threat_info in self.threats.items():
            match = True
            for key, value in query.items():
                if key not in threat_info or threat_info[key] != value:
                    match = False
                    break
            
            if match:
                result = threat_info.copy()
                result["id"] = threat_id
                results.append(result)
        
        return results

# Example usage (for demonstration)
if __name__ == "__main__":
    # Create a knowledge base
    kb = MemoryKnowledgeBase("Demo Knowledge Base")
    kb.store("admin_user", {"username": "admin", "access_level": "full"})
    kb.store("known_ips", ["192.168.1.1", "10.0.0.1", "172.16.0.1"])
    
    # Create a pattern recognizer with some patterns
    recognizer = PatternRecognizer()
    
    # Create a pattern for unauthorized access attempts
    unauthorized_pattern = SimplePattern(
        "unauth_access",
        "Unauthorized Access Attempt",
        "Detects attempts to access restricted resources without authorization"
    )
    unauthorized_pattern.add_required_match("access_attempt", True)
    unauthorized_pattern.add_required_match("authorized", False)
    unauthorized_pattern.set_threshold(0.8)
    
    # Register the pattern
    recognizer.register_pattern(unauthorized_pattern)
    
    # Test pattern recognition
    test_data = {
        "timestamp": time.time(),
        "access_attempt": True,
        "authorized": False,
        "resource": "admin_panel",
        "ip_address": "198.51.100.1"
    }
    
    matches = recognizer.recognize(test_data)
    
    for pattern, confidence in matches:
        print(f"Matched pattern: {pattern.name} (confidence: {confidence:.2f})")
    
    # Create an anomaly detector
    detector = AnomalyDetector()
    
    # Add some baseline data
    for i in range(100):
        detector.add_data_point("cpu_usage", random.uniform(10, 30))
    
    # Test anomaly detection
    normal_value = 25.0
    anomaly_value = 95.0
    
    is_normal_anomaly, normal_score = detector.is_anomaly("cpu_usage", normal_value)
    is_high_anomaly, high_score = detector.is_anomaly("cpu_usage", anomaly_value)
    
    print(f"Normal value ({normal_value}) is anomaly: {is_normal_anomaly} (score: {normal_score:.2f})")
    print(f"High value ({anomaly_value}) is anomaly: {is_high_anomaly} (score: {high_score:.2f})")
