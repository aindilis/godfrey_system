# godfrey_system/main.py
"""
Godfrey System - Main Application
--------------------------------
Entry point for the Godfrey vigilance and strategic intelligence system.
"""

import argparse
import logging
import os
import time
import json
import threading
import signal
import sys
from typing import Dict, List, Any, Tuple

# Import Godfrey system modules
from godfrey_system.core import (
    GodfredEngine, AlertLevel, Action, Threat, Observation, CorePrinciples
)
from godfrey_system.sensors import (
    SensorManager, MotionSensor, NetworkMonitor, TextAnalyzer, SecurityCameraSensor
)
from godfrey_system.knowledge import (
    MemoryKnowledgeBase, FileKnowledgeBase, PatternRecognizer, 
    SimplePattern, SequencePattern, AnomalyDetector
)
from godfrey_system.agents import (
    AgentSystem, VigilanceAgent, DecisionAgent, MessageType, Message
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("godfrey.log")
    ]
)

logger = logging.getLogger("Godfrey")

class GodfredSystem:
    """Main class that integrates all Godfrey components"""
    
    def __init__(self, config_file=None):
        logger.info("Initializing Godfrey System")
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize core components
        self.engine = GodfredEngine()
        self.sensor_manager = SensorManager()
        self.agent_system = AgentSystem.instance()
        
        # Initialize knowledge base
        kb_type = self.config.get("knowledge_base", {}).get("type", "memory")
        kb_path = self.config.get("knowledge_base", {}).get("path", "godfrey_knowledge.json")
        
        if kb_type == "file":
            self.knowledge_base = FileKnowledgeBase("Godfrey Knowledge", kb_path)
        else:
            self.knowledge_base = MemoryKnowledgeBase("Godfrey Knowledge")
        
        # Initialize pattern recognizer
        self.pattern_recognizer = PatternRecognizer()
        
        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector()
        
        # Flag for signaling system shutdown
        self.running = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_file):
        """Load configuration from file or use defaults"""
        config = {
            "name": "Godfrey System",
            "version": "1.0.0",
            "knowledge_base": {
                "type": "memory",
                "path": "godfrey_knowledge.json"
            },
            "sensors": {
                "poll_interval": 5  # seconds
            },
            "vigilance": {
                "alert_threshold": 70  # 0-100
            },
            "principles": {
                "vigilance_weight": 1.5,
                "resilience_weight": 1.3,
                "foresight_weight": 1.2,
                "adaptability_weight": 1.1,
                "integrity_weight": 1.0
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Update with loaded config
                self._deep_update(config, loaded_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return config
    
    def _deep_update(self, d, u):
        """Recursively update a dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
    
    def _signal_handler(self, sig, frame):
        """Handle signals for graceful shutdown"""
        logger.info("Received shutdown signal")
        self.stop()
    
    def setup(self):
        """Set up the system components"""
        logger.info("Setting up Godfrey System components")
        
        # Set up principles with weights from config
        principles_config = self.config.get("principles", {})
        for name, principle in self.engine.principles.principles.items():
            weight_key = f"{name}_weight"
            if weight_key in principles_config:
                principle.weight = principles_config[weight_key]
        
        # Set up sensors based on configuration or defaults
        self._setup_sensors()
        
        # Set up threat patterns
        self._setup_threat_patterns()
        
        # Set up action templates
        self._setup_action_templates()
        
        # Set up agents
        self._setup_agents()
        
        # Connect sensors to engine
        self._connect_sensors_to_engine()
        
        logger.info("Godfrey System setup complete")
    
    def _setup_sensors(self):
        """Set up sensors based on configuration"""
        # Create and register some default sensors
        
        # Motion sensor
        motion_sensor = MotionSensor("motion1", "Main Entrance Motion", "Primary Entrance")
        self.sensor_manager.register_sensor(motion_sensor)
        
        # Network monitor
        network_monitor = NetworkMonitor("network1", "Primary Network Monitor")
        self.sensor_manager.register_sensor(network_monitor)
        
        # Text analyzer
        text_analyzer = TextAnalyzer("text1", "Communication Analyzer")
        text_analyzer.configure(
            keywords=["urgent", "alert", "warning", "breach", "compromise", "suspicious", "unusual"],
            sentiment_analysis=True
        )
        self.sensor_manager.register_sensor(text_analyzer)
        
        # Security camera
        security_camera = SecurityCameraSensor("camera1", "Primary Security Camera", "Main Entrance")
        self.sensor_manager.register_sensor(security_camera)
        
        logger.info(f"Registered {len(self.sensor_manager.sensors)} sensors")
    
    def _setup_threat_patterns(self):
        """Set up threat patterns for detection"""
        # Unauthorized access pattern
        unauthorized_pattern = SimplePattern(
            "unauthorized_access",
            "Unauthorized Access Attempt",
            "Detects attempts to access restricted resources without authorization"
        )
        unauthorized_pattern.add_required_match("access_attempt", True)
        unauthorized_pattern.add_required_match("authorized", False)
        unauthorized_pattern.set_threshold(0.8)
        self.pattern_recognizer.register_pattern(unauthorized_pattern)
        
        # Suspicious network activity pattern
        network_pattern = SimplePattern(
            "suspicious_network",
            "Suspicious Network Activity",
            "Detects unusual network traffic patterns that may indicate an attack"
        )
        network_pattern.add_required_match("suspicious_activity", True)
        network_pattern.add_optional_match("unknown_ip", None)  # Any non-None value
        network_pattern.set_threshold(0.6)
        self.pattern_recognizer.register_pattern(network_pattern)
        
        # Unusual motion pattern
        motion_pattern = SimplePattern(
            "unusual_motion",
            "Unusual Motion Detection",
            "Detects motion in areas that should be unoccupied"
        )
        motion_pattern.add_required_match("motion_detected", True)
        motion_pattern.set_threshold(0.7)
        self.pattern_recognizer.register_pattern(motion_pattern)
        
        # Negative sentiment communication pattern
        sentiment_pattern = SimplePattern(
            "negative_sentiment",
            "Negative Communication Sentiment",
            "Detects unusually negative sentiment in communications"
        )
        sentiment_pattern.add_optional_match("average_sentiment", -0.5)  # This uses a threshold check in the pattern
        sentiment_pattern.set_threshold(0.6)
        self.pattern_recognizer.register_pattern(sentiment_pattern)
        
        # Add an escalation sequence pattern
        escalation_pattern = SequencePattern(
            "security_escalation",
            "Security Incident Escalation",
            "Detects a sequence of events that indicate an escalating security incident"
        )
        escalation_pattern.add_sequence_event({"pattern": "unauthorized_access"})
        escalation_pattern.add_sequence_event({"pattern": "suspicious_network"})
        escalation_pattern.set_max_time_gap(300)  # 5 minutes
        self.pattern_recognizer.register_pattern(escalation_pattern)
        
        logger.info(f"Registered {len(self.pattern_recognizer.patterns)} threat patterns")
    
    def _setup_action_templates(self):
        """Set up action templates for response"""
        # Add action templates to the engine
        
        # Lockdown action
        self.engine.action_generator.add_action_template(
            "lockdown_systems",
            {
                "name": "Lock Down Systems",
                "description": "Temporarily restrict all access to critical systems",
                "applicable_threats": ["unauthorized_access", "suspicious_network", "security_escalation"],
                "base_effectiveness": 0.8,
                "base_resource_cost": 0.7,
                "principles_alignment": {
                    "vigilance": 0.9,
                    "resilience": 0.7,
                    "adaptability": -0.3,
                    "foresight": 0.5,
                    "integrity": 0.6
                }
            }
        )
        
        # Alert security action
        self.engine.action_generator.add_action_template(
            "alert_security",
            {
                "name": "Alert Security Team",
                "description": "Send immediate alert to security personnel",
                "applicable_threats": ["unauthorized_access", "unusual_motion", "security_escalation"],
                "base_effectiveness": 0.7,
                "base_resource_cost": 0.2,
                "principles_alignment": {
                    "vigilance": 0.8,
                    "resilience": 0.5,
                    "adaptability": 0.6,
                    "foresight": 0.4,
                    "integrity": 0.9
                }
            }
        )
        
        # Increase monitoring action
        self.engine.action_generator.add_action_template(
            "increase_monitoring",
            {
                "name": "Increase Monitoring",
                "description": "Increase the sensitivity and frequency of monitoring systems",
                "applicable_threats": ["suspicious_network", "negative_sentiment"],
                "base_effectiveness": 0.6,
                "base_resource_cost": 0.4,
                "principles_alignment": {
                    "vigilance": 1.0,
                    "resilience": 0.3,
                    "adaptability": 0.7,
                    "foresight": 0.8,
                    "integrity": 0.5
                }
            }
        )
        
        # Backup critical data action
        self.engine.action_generator.add_action_template(
            "backup_data",
            {
                "name": "Backup Critical Data",
                "description": "Create emergency backups of all critical data",
                "applicable_threats": ["security_escalation"],
                "base_effectiveness": 0.9,
                "base_resource_cost": 0.5,
                "principles_alignment": {
                    "vigilance": 0.6,
                    "resilience": 1.0,
                    "adaptability": 0.4,
                    "foresight": 0.9,
                    "integrity": 0.7
                }
            }
        )
        
        logger.info(f"Registered {len(self.engine.action_generator.action_templates)} action templates")
    
    def _setup_agents(self):
        """Set up the agent system"""
        # Create a vigilance agent
        vigilance_agent = VigilanceAgent("vigilance1", "Primary Vigilance Agent")
        
        # Add threat patterns to the vigilance agent
        vigilance_agent.add_threat_pattern(
            "unauthorized_access",
            {
                "name": "Unauthorized Access Attempt",
                "description": "Detects attempts to access restricted resources without authorization",
                "indicators": {
                    "access_attempt": True,
                    "authorized": False
                }
            }
        )
        
        vigilance_agent.add_threat_pattern(
            "suspicious_network",
            {
                "name": "Suspicious Network Activity",
                "description": "Detects unusual network traffic patterns that may indicate an attack",
                "indicators": {
                    "suspicious_activity": True
                }
            }
        )
        
        # Set alert threshold from config
        alert_threshold = self.config.get("vigilance", {}).get("alert_threshold", 70)
        vigilance_agent.set_alert_threshold(alert_threshold)
        
        # Create a decision agent
        decision_agent = DecisionAgent("decision1", "Primary Decision Agent")
        
        # Add action templates to the decision agent
        decision_agent.add_action_template(
            "lockdown_systems",
            {
                "name": "Lock Down Systems",
                "description": "Temporarily restrict all access to critical systems",
                "applicable_alerts": ["unauthorized_access", "suspicious_network"],
                "base_effectiveness": 0.8,
                "base_resource_cost": 0.7
            }
        )
        
        decision_agent.add_action_template(
            "alert_security",
            {
                "name": "Alert Security Team",
                "description": "Send immediate alert to security personnel",
                "applicable_alerts": ["unauthorized_access", "suspicious_network"],
                "base_effectiveness": 0.7,
                "base_resource_cost": 0.2
            }
        )
        
        # Register agents with the system
        self.agent_system.register_agent(vigilance_agent)
        self.agent_system.register_agent(decision_agent)
        
        logger.info(f"Registered {len(self.agent_system.agents)} agents")
    
    def _connect_sensors_to_engine(self):
        """Connect sensors to the Godfrey engine"""
        # Register callbacks for sensor readings
        for sensor_id, sensor in self.sensor_manager.sensors.items():
            self.sensor_manager.register_callback(
                sensor_id,
                lambda reading: self._sensor_callback(reading)
            )
    
    def _sensor_callback(self, reading):
        """Callback for sensor readings"""
        # Create an observation from the sensor reading
        observation = Observation(
            reading.sensor_id,
            reading.timestamp,
            reading.data
        )
        
        # Add to the engine
        self.engine.situation_awareness.add_observation(observation)
        
        # Also send to the vigilance agent if it exists
        vigilance_agent = self.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender=reading.sensor_id,
                receiver="vigilance1",
                message_type=MessageType.EVENT,
                content=reading.data,
                metadata={"reading_id": reading.sensor_id, "timestamp": reading.timestamp}
            ))
    
    def start(self):
        """Start the Godfrey system"""
        if self.running:
            logger.warning("Godfrey System is already running")
            return
        
        logger.info("Starting Godfrey System")
        self.running = True
        
        # Start agent system
        self.agent_system.start_all_agents()
        
        # Start sensors
        self.sensor_manager.start_all_sensors()
        
        # Start main processing loop
        self._start_processing_loop()
        
        logger.info("Godfrey System started")
    
    def stop(self):
        """Stop the Godfrey system"""
        if not self.running:
            return
        
        logger.info("Stopping Godfrey System")
        self.running = False
        
        # Stop sensors
        self.sensor_manager.stop_all_sensors()
        
        # Stop agent system
        self.agent_system.stop_all_agents()
        
        logger.info("Godfrey System stopped")
    
    def _start_processing_loop(self):
        """Start the main processing loop in a background thread"""
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _processing_loop(self):
        """Main processing loop"""
        poll_interval = self.config.get("sensors", {}).get("poll_interval", 5)
        
        while self.running:
            try:
                # Get sensor readings
                readings = self.sensor_manager.get_readings()
                
                # Orient based on recent observations
                threats = self.engine.orient()
                
                # Generate actions based on threats
                if threats:
                    actions = self.engine.decide(threats)
                    
                    # Log recommended actions
                    if actions:
                        for action in actions[:3]:  # Top 3 actions
                            logger.info(f"Recommended action: {action.name} "
                                       f"(effectiveness: {action.effectiveness:.2f}, "
                                       f"cost: {action.resource_cost:.2f})")
                
                # Sleep for the poll interval
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1)  # Sleep briefly to avoid tight loop on error

def simulate_scenario(system):
    """Simulate a security scenario for demonstration purposes"""
    logger.info("Starting scenario simulation")
    
    # Get sensors
    motion_sensor = system.sensor_manager.get_sensor("motion1")
    network_monitor = system.sensor_manager.get_sensor("network1")
    text_analyzer = system.sensor_manager.get_sensor("text1")
    
    # Simulate a timeline of events
    
    # First, everything is normal
    logger.info("Scenario: Normal operations")
    time.sleep(5)
    
    # Simulate suspicious network activity
    logger.info("Scenario: Suspicious network activity detected")
    network_monitor.receive_message(Message(
        sender="simulation",
        receiver=network_monitor.id,
        message_type=MessageType.EVENT,
        content={
            "traffic_level": 0.8,
            "suspicious_activity": True,
            "unknown_ip": "203.0.113.100",
            "anomaly_score": 0.75,
            "alert": True
        }
    ))
    time.sleep(5)
    
    # Simulate motion detection
    logger.info("Scenario: Motion detected in secure area")
    motion_sensor.receive_message(Message(
        sender="simulation",
        receiver=motion_sensor.id,
        message_type=MessageType.EVENT,
        content={
            "motion_detected": True,
            "location": "Server Room"
        }
    ))
    time.sleep(5)
    
    # Simulate suspicious communication
    logger.info("Scenario: Suspicious communication detected")
    text_analyzer.add_text(
        "URGENT: System compromised. Need immediate access to admin credentials.",
        "email"
    )
    time.sleep(10)
    
    logger.info("Scenario simulation complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Godfrey Vigilance System")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--simulate", "-s", action="store_true", help="Run a simulated scenario")
    args = parser.parse_args()
    
    # Create and set up the system
    system = GodfredSystem(args.config)
    system.setup()
    
    try:
        # Start the system
        system.start()
        
        # Run a simulated scenario if requested
        if args.simulate:
            simulate_scenario(system)
        
        # Keep running until interrupted
        while system.running:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Stop the system
        system.stop()

if __name__ == "__main__":
    main()
    