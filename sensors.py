# godfrey_system/sensors.py
"""
Godfrey System - Sensor Interface
--------------------------------
Interface for physical and virtual sensors that feed data into the Godfrey system.
"""

import time
import random
import threading
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
import logging

logger = logging.getLogger("Godfrey.Sensors")

class SensorType(Enum):
    """Types of sensors supported by the system"""
    PHYSICAL = 1     # Hardware sensors
    VIRTUAL = 2      # Software monitors
    SEMANTIC = 3     # Natural language/content analysis
    COMPOSITE = 4    # Combination of multiple sensors

class SensorStatus(Enum):
    """Status of a sensor"""
    OFFLINE = 0
    ONLINE = 1
    DEGRADED = 2
    ERROR = 3

class SensorReading:
    """A reading from a sensor"""
    
    def __init__(self, sensor_id: str, timestamp: float, data: Dict[str, Any], metadata: Dict[str, Any] = None):
        self.sensor_id = sensor_id
        self.timestamp = timestamp
        self.data = data
        self.metadata = metadata or {}
    
    def __str__(self):
        return f"SensorReading({self.sensor_id}, {self.timestamp}, {self.data})"

class Sensor:
    """Base class for all sensors"""
    
    def __init__(self, sensor_id: str, name: str, sensor_type: SensorType, description: str = ""):
        self.id = sensor_id
        self.name = name
        self.type = sensor_type
        self.description = description
        self.status = SensorStatus.OFFLINE
        self.last_reading = None
        self.last_error = None
        self.metadata = {}
    
    def get_reading(self) -> Optional[SensorReading]:
        """Get a reading from the sensor"""
        raise NotImplementedError("Subclasses must implement get_reading()")
    
    def start(self):
        """Start the sensor (e.g., connect to hardware, initialize)"""
        self.status = SensorStatus.ONLINE
        logger.info(f"Started sensor: {self.name} ({self.id})")
    
    def stop(self):
        """Stop the sensor"""
        self.status = SensorStatus.OFFLINE
        logger.info(f"Stopped sensor: {self.name} ({self.id})")

class PhysicalSensor(Sensor):
    """A sensor that interfaces with physical hardware"""
    
    def __init__(self, sensor_id: str, name: str, description: str = ""):
        super().__init__(sensor_id, name, SensorType.PHYSICAL, description)
        self.hardware_id = None
        self.connection_params = {}
    
    def connect(self, hardware_id: str, **connection_params):
        """Connect to the physical sensor"""
        self.hardware_id = hardware_id
        self.connection_params = connection_params
        # In a real implementation, this would connect to actual hardware
        self.status = SensorStatus.ONLINE
        logger.info(f"Connected to physical sensor: {self.name} ({hardware_id})")
    
    def disconnect(self):
        """Disconnect from the physical sensor"""
        # In a real implementation, this would disconnect from hardware
        self.status = SensorStatus.OFFLINE
        logger.info(f"Disconnected from physical sensor: {self.name}")

class VirtualSensor(Sensor):
    """A sensor that monitors software or virtual systems"""
    
    def __init__(self, sensor_id: str, name: str, description: str = ""):
        super().__init__(sensor_id, name, SensorType.VIRTUAL, description)
        self.monitor_interval = 60  # seconds
        self.monitor_thread = None
        self.should_run = False
        self.callback = None
    
    def start_monitoring(self, interval: int = None, callback: Callable = None):
        """Start monitoring with the specified interval"""
        if interval:
            self.monitor_interval = interval
        
        self.callback = callback
        self.should_run = True
        self.status = SensorStatus.ONLINE
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"Started monitoring with virtual sensor: {self.name} (interval: {self.monitor_interval}s)")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.should_run = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        self.status = SensorStatus.OFFLINE
        logger.info(f"Stopped monitoring with virtual sensor: {self.name}")
    
    def _monitoring_loop(self):
        """Internal loop for monitoring"""
        while self.should_run:
            try:
                reading = self.get_reading()
                self.last_reading = reading
                
                if self.callback and reading:
                    self.callback(reading)
                
            except Exception as e:
                self.last_error = str(e)
                self.status = SensorStatus.ERROR
                logger.error(f"Error in virtual sensor {self.name}: {e}")
            
            # Sleep until next interval
            time.sleep(self.monitor_interval)

class SemanticSensor(VirtualSensor):
    """A sensor that analyzes text, documents, or communication"""
    
    def __init__(self, sensor_id: str, name: str, description: str = ""):
        super().__init__(sensor_id, name, description)
        self.type = SensorType.SEMANTIC
        self.sources = []
        self.keywords = []
        self.sentiment_analysis = False
    
    def configure(self, sources: List[str] = None, keywords: List[str] = None, sentiment_analysis: bool = False):
        """Configure the semantic sensor"""
        if sources:
            self.sources = sources
        if keywords:
            self.keywords = keywords
        self.sentiment_analysis = sentiment_analysis
        
        logger.info(f"Configured semantic sensor: {self.name} (sources: {len(self.sources)}, keywords: {len(self.keywords)})")

class CompositeSensor(Sensor):
    """A sensor that combines readings from multiple other sensors"""
    
    def __init__(self, sensor_id: str, name: str, description: str = ""):
        super().__init__(sensor_id, name, SensorType.COMPOSITE, description)
        self.child_sensors = {}  # id -> sensor
        self.aggregation_function = None
    
    def add_child_sensor(self, sensor: Sensor):
        """Add a child sensor"""
        self.child_sensors[sensor.id] = sensor
        logger.info(f"Added child sensor {sensor.name} to composite sensor {self.name}")
    
    def remove_child_sensor(self, sensor_id: str):
        """Remove a child sensor"""
        if sensor_id in self.child_sensors:
            del self.child_sensors[sensor_id]
            logger.info(f"Removed child sensor {sensor_id} from composite sensor {self.name}")
    
    def set_aggregation_function(self, func: Callable):
        """Set the function used to aggregate readings from child sensors"""
        self.aggregation_function = func
    
    def get_reading(self) -> Optional[SensorReading]:
        """Get an aggregated reading from all child sensors"""
        if not self.child_sensors:
            return None
        
        # Collect readings from all child sensors
        readings = {}
        for sensor_id, sensor in self.child_sensors.items():
            if sensor.status == SensorStatus.ONLINE:
                reading = sensor.get_reading()
                if reading:
                    readings[sensor_id] = reading
        
        if not readings:
            return None
        
        # If no aggregation function is set, use the default aggregation
        if not self.aggregation_function:
            return self._default_aggregation(readings)
        
        # Use the custom aggregation function
        try:
            aggregated_data = self.aggregation_function(readings)
            return SensorReading(
                sensor_id=self.id,
                timestamp=time.time(),
                data=aggregated_data,
                metadata={"source_readings": readings}
            )
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in composite sensor aggregation: {e}")
            return None
    
    def _default_aggregation(self, readings: Dict[str, SensorReading]) -> SensorReading:
        """Default aggregation method that merges all data fields"""
        aggregated_data = {}
        for sensor_id, reading in readings.items():
            # Use sensor ID as prefix to avoid key collisions
            for key, value in reading.data.items():
                aggregated_data[f"{sensor_id}.{key}"] = value
        
        return SensorReading(
            sensor_id=self.id,
            timestamp=time.time(),
            data=aggregated_data,
            metadata={"source_readings": [r.sensor_id for r in readings.values()]}
        )

# Sample sensor implementations

class MotionSensor(PhysicalSensor):
    """A sensor that detects motion in an area"""
    
    def __init__(self, sensor_id: str, name: str = "Motion Sensor", location: str = "Unknown"):
        super().__init__(sensor_id, name, f"Motion sensor located at {location}")
        self.location = location
        self.sensitivity = 0.5  # 0.0 to 1.0
    
    def get_reading(self) -> SensorReading:
        """Get a reading from the motion sensor"""
        # In a real implementation, this would interface with hardware
        # Here we simulate a motion detection with some probability
        motion_detected = random.random() < self.sensitivity * 0.2  # 20% chance when at max sensitivity
        
        return SensorReading(
            sensor_id=self.id,
            timestamp=time.time(),
            data={
                "motion_detected": motion_detected,
                "location": self.location
            },
            metadata={
                "sensitivity": self.sensitivity
            }
        )

class NetworkMonitor(VirtualSensor):
    """A sensor that monitors network traffic and security"""
    
    def __init__(self, sensor_id: str, name: str = "Network Monitor", interface: str = "all"):
        super().__init__(sensor_id, name, f"Network traffic monitor for interface {interface}")
        self.interface = interface
        self.alert_threshold = 0.7  # 0.0 to 1.0
        self.known_ips = set()
    
    def get_reading(self) -> SensorReading:
        """Get a reading from the network monitor"""
        # In a real implementation, this would analyze actual network traffic
        # Here we simulate network activity
        
        # Simulate traffic levels
        traffic_level = random.random()  # 0.0 to 1.0
        
        # Simulate suspicious activity with some probability
        suspicious_activity = random.random() < 0.15  # 15% chance
        
        # Simulate an unknown IP with some probability
        unknown_ip = None
        if suspicious_activity and random.random() < 0.5:
            # Generate a random IP
            unknown_ip = f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
        
        # Calculate an anomaly score
        anomaly_score = 0.0
        if suspicious_activity:
            anomaly_score = random.uniform(0.4, 0.9)
        
        # Determine if this should trigger an alert
        alert = anomaly_score > self.alert_threshold
        
        return SensorReading(
            sensor_id=self.id,
            timestamp=time.time(),
            data={
                "traffic_level": traffic_level,
                "suspicious_activity": suspicious_activity,
                "unknown_ip": unknown_ip,
                "anomaly_score": anomaly_score,
                "alert": alert
            },
            metadata={
                "interface": self.interface,
                "alert_threshold": self.alert_threshold
            }
        )

class TextAnalyzer(SemanticSensor):
    """A sensor that analyzes text for concerning patterns"""
    
    def __init__(self, sensor_id: str, name: str = "Text Analyzer"):
        super().__init__(sensor_id, name, "Analyzes text for concerning patterns or keywords")
        self.recent_texts = []
        self.max_texts = 100
    
    def add_text(self, text: str, source: str):
        """Add a text to be analyzed"""
        self.recent_texts.append({
            "text": text,
            "source": source,
            "timestamp": time.time()
        })
        
        # Keep only the most recent texts
        if len(self.recent_texts) > self.max_texts:
            self.recent_texts = self.recent_texts[-self.max_texts:]
    
    def get_reading(self) -> Optional[SensorReading]:
        """Analyze the recent texts for keywords and sentiment"""
        if not self.recent_texts:
            return None
        
        # Only analyze texts from the last hour
        cutoff_time = time.time() - 3600
        recent_texts = [t for t in self.recent_texts if t["timestamp"] >= cutoff_time]
        
        if not recent_texts:
            return None
        
        # In a real implementation, this would use NLP techniques
        # Here we simulate keyword detection and sentiment analysis
        
        # Check for keyword matches
        keyword_matches = []
        for text_entry in recent_texts:
            text = text_entry["text"].lower()
            matches = [kw for kw in self.keywords if kw.lower() in text]
            if matches:
                keyword_matches.append({
                    "source": text_entry["source"],
                    "timestamp": text_entry["timestamp"],
                    "keywords": matches
                })
        
        # Simulate sentiment analysis
        sentiment_scores = []
        if self.sentiment_analysis:
            for text_entry in recent_texts:
                # Simulate sentiment score between -1.0 (negative) and 1.0 (positive)
                sentiment = random.uniform(-1.0, 1.0)
                sentiment_scores.append({
                    "source": text_entry["source"],
                    "timestamp": text_entry["timestamp"],
                    "sentiment": sentiment
                })
        
        # Calculate average sentiment
        avg_sentiment = 0.0
        if sentiment_scores:
            avg_sentiment = sum(s["sentiment"] for s in sentiment_scores) / len(sentiment_scores)
        
        return SensorReading(
            sensor_id=self.id,
            timestamp=time.time(),
            data={
                "keyword_matches": keyword_matches,
                "sentiment_scores": sentiment_scores,
                "average_sentiment": avg_sentiment,
                "num_texts_analyzed": len(recent_texts)
            },
            metadata={
                "keywords": self.keywords,
                "sentiment_analysis": self.sentiment_analysis
            }
        )

class SecurityCameraSensor(PhysicalSensor):
    """A sensor that interfaces with security cameras"""
    
    def __init__(self, sensor_id: str, name: str, location: str):
        super().__init__(sensor_id, name, f"Security camera at {location}")
        self.location = location
        self.motion_detection_enabled = True
        self.face_recognition_enabled = False
        self.recognized_faces = set()
    
    def get_reading(self) -> SensorReading:
        """Get a reading from the security camera"""
        # In a real implementation, this would interface with an actual camera
        # Here we simulate camera events
        
        # Simulate motion detection
        motion_detected = False
        if self.motion_detection_enabled:
            motion_detected = random.random() < 0.2  # 20% chance
        
        # Simulate face recognition
        face_detected = False
        recognized_face = None
        if self.face_recognition_enabled and motion_detected:
            face_detected = random.random() < 0.5  # 50% chance if motion detected
            if face_detected and self.recognized_faces:
                # Randomly select a known face or generate an unknown one
                if random.random() < 0.7:  # 70% chance of recognized face
                    recognized_face = random.choice(list(self.recognized_faces))
        
        return SensorReading(
            sensor_id=self.id,
            timestamp=time.time(),
            data={
                "motion_detected": motion_detected,
                "face_detected": face_detected,
                "recognized_face": recognized_face,
                "location": self.location
            },
            metadata={
                "motion_detection_enabled": self.motion_detection_enabled,
                "face_recognition_enabled": self.face_recognition_enabled
            }
        )

class SensorManager:
    """Manages a collection of sensors"""
    
    def __init__(self):
        self.sensors = {}  # id -> sensor
        self.callbacks = {}  # sensor_id -> [callbacks]
    
    def register_sensor(self, sensor: Sensor):
        """Register a sensor with the manager"""
        self.sensors[sensor.id] = sensor
        logger.info(f"Registered sensor: {sensor.name} ({sensor.id})")
    
    def unregister_sensor(self, sensor_id: str):
        """Unregister a sensor"""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            if sensor_id in self.callbacks:
                del self.callbacks[sensor_id]
            logger.info(f"Unregistered sensor: {sensor_id}")
    
    def register_callback(self, sensor_id: str, callback: Callable):
        """Register a callback for a sensor's readings"""
        if sensor_id not in self.callbacks:
            self.callbacks[sensor_id] = []
        
        self.callbacks[sensor_id].append(callback)
        logger.debug(f"Registered callback for sensor: {sensor_id}")
    
    def unregister_callback(self, sensor_id: str, callback: Callable):
        """Unregister a callback"""
        if sensor_id in self.callbacks and callback in self.callbacks[sensor_id]:
            self.callbacks[sensor_id].remove(callback)
            logger.debug(f"Unregistered callback for sensor: {sensor_id}")
    
    def get_sensor(self, sensor_id: str) -> Optional[Sensor]:
        """Get a sensor by ID"""
        return self.sensors.get(sensor_id)
    
    def get_all_sensors(self) -> List[Sensor]:
        """Get all registered sensors"""
        return list(self.sensors.values())
    
    def get_sensors_by_type(self, sensor_type: SensorType) -> List[Sensor]:
        """Get all sensors of a specific type"""
        return [s for s in self.sensors.values() if s.type == sensor_type]
    
    def start_all_sensors(self):
        """Start all registered sensors"""
        for sensor in self.sensors.values():
            try:
                sensor.start()
            except Exception as e:
                logger.error(f"Error starting sensor {sensor.name}: {e}")
    
    def stop_all_sensors(self):
        """Stop all registered sensors"""
        for sensor in self.sensors.values():
            try:
                sensor.stop()
            except Exception as e:
                logger.error(f"Error stopping sensor {sensor.name}: {e}")
    
    def get_readings(self) -> Dict[str, SensorReading]:
        """Get readings from all sensors"""
        readings = {}
        for sensor_id, sensor in self.sensors.items():
            if sensor.status == SensorStatus.ONLINE:
                try:
                    reading = sensor.get_reading()
                    if reading:
                        readings[sensor_id] = reading
                        # Trigger callbacks if any
                        if sensor_id in self.callbacks:
                            for callback in self.callbacks[sensor_id]:
                                try:
                                    callback(reading)
                                except Exception as e:
                                    logger.error(f"Error in callback for sensor {sensor_id}: {e}")
                except Exception as e:
                    logger.error(f"Error getting reading from sensor {sensor.name}: {e}")
        
        return readings

# Example usage (for demonstration)
if __name__ == "__main__":
    # Create sensor manager
    manager = SensorManager()
    
    # Create and register some sensors
    motion_sensor = MotionSensor("motion1", "Main Entrance Motion", "Building A Entrance")
    network_monitor = NetworkMonitor("network1", "Primary Network Monitor")
    text_analyzer = TextAnalyzer("text1")
    text_analyzer.configure(keywords=["urgent", "alert", "warning", "breach"])
    
    manager.register_sensor(motion_sensor)
    manager.register_sensor(network_monitor)
    manager.register_sensor(text_analyzer)
    
    # Start all sensors
    manager.start_all_sensors()
    
    # Add some text for analysis
    text_analyzer.add_text("There may be a potential security breach in sector 7", "email")
    text_analyzer.add_text("Just a routine system check, all clear", "chat")
    
    # Get readings from all sensors
    readings = manager.get_readings()
    
    # Print readings
    for sensor_id, reading in readings.items():
        print(f"Reading from {sensor_id}: {reading}")
    
    # Stop all sensors
    manager.stop_all_sensors()