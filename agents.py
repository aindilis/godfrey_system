# godfrey_system/agents.py
"""
Godfrey System - Agent Framework
-------------------------------
Multi-agent system architecture inspired by the UniLang Agent framework.
This provides a flexible way for different components to communicate and cooperate.
"""

import logging
import time
import uuid
import threading
import queue
from typing import Dict, List, Any, Callable, Optional, Set
from enum import Enum

logger = logging.getLogger("Godfrey.Agents")

class MessageType(Enum):
    """Types of messages that can be exchanged between agents"""
    REQUEST = 1
    RESPONSE = 2
    ALERT = 3
    INFO = 4
    COMMAND = 5
    EVENT = 6

class Message:
    """A message that can be passed between agents"""
    
    def __init__(self, 
                 sender: str, 
                 receiver: str, 
                 message_type: MessageType, 
                 content: Any,
                 correlation_id: str = None,
                 metadata: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.type = message_type
        self.content = content
        self.correlation_id = correlation_id or self.id
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def __str__(self):
        return f"Message({self.type.name}, from={self.sender}, to={self.receiver})"
    
    def create_response(self, content: Any) -> 'Message':
        """Create a response to this message"""
        return Message(
            sender=self.receiver,
            receiver=self.sender,
            message_type=MessageType.RESPONSE,
            content=content,
            correlation_id=self.correlation_id,
            metadata={
                "in_response_to": self.id,
                "original_type": self.type.name
            }
        )

class Agent:
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, name: str):
        self.id = agent_id
        self.name = name
        self.inbox = queue.Queue()
        self.subscriptions = set()  # Topics this agent is subscribed to
        self.message_handlers = {}  # message_type -> handler_function
        self.running = False
        self.processing_thread = None
        self.startup_time = None
        self.message_count = 0
    
    def start(self):
        """Start the agent"""
        if self.running:
            return
        
        self.running = True
        self.startup_time = time.time()
        self.processing_thread = threading.Thread(target=self._process_messages)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info(f"Started agent: {self.name} ({self.id})")
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        logger.info(f"Stopped agent: {self.name} ({self.id})")
    
    def receive_message(self, message: Message):
        """Receive a message"""
        self.inbox.put(message)
    
    def send_message(self, receiver: str, message_type: MessageType, content: Any, 
                    correlation_id: str = None, metadata: Dict[str, Any] = None) -> Message:
        """Create and send a message to another agent"""
        message = Message(
            sender=self.id,
            receiver=receiver,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
            metadata=metadata
        )
        
        AgentSystem.instance().send_message(message)
        return message
    
    def register_handler(self, message_type: MessageType, handler: Callable[[Message], None]):
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler
        logger.debug(f"Agent {self.name} registered handler for {message_type.name}")
    
    def subscribe(self, topic: str):
        """Subscribe to a topic"""
        self.subscriptions.add(topic)
        AgentSystem.instance().subscribe(self.id, topic)
        logger.debug(f"Agent {self.name} subscribed to {topic}")
    
    def unsubscribe(self, topic: str):
        """Unsubscribe from a topic"""
        if topic in self.subscriptions:
            self.subscriptions.remove(topic)
            AgentSystem.instance().unsubscribe(self.id, topic)
            logger.debug(f"Agent {self.name} unsubscribed from {topic}")
    
    def publish(self, topic: str, content: Any, metadata: Dict[str, Any] = None) -> Message:
        """Publish content to a topic"""
        return AgentSystem.instance().publish(self.id, topic, content, metadata)
    
    def _process_messages(self):
        """Process messages in the inbox"""
        while self.running:
            try:
                # Get message with a timeout to allow checking if we're still running
                try:
                    message = self.inbox.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                self.message_count += 1
                
                # Handle the message
                try:
                    self._handle_message(message)
                except Exception as e:
                    logger.error(f"Error handling message in agent {self.name}: {e}")
                
                # Mark as done
                self.inbox.task_done()
                
            except Exception as e:
                logger.error(f"Error in message processing loop for agent {self.name}: {e}")
    
    def _handle_message(self, message: Message):
        """Handle a single message"""
        # Check if we have a specific handler for this message type
        if message.type in self.message_handlers:
            self.message_handlers[message.type](message)
            return
        
        # Default handling
        self._default_message_handler(message)
    
    def _default_message_handler(self, message: Message):
        """Default message handler if no specific handler is registered"""
        logger.debug(f"Agent {self.name} has no handler for {message.type.name} message from {message.sender}")
        
        # For requests, send a default response
        if message.type == MessageType.REQUEST:
            response = message.create_response({
                "status": "error",
                "message": f"No handler registered for {message.type.name}"
            })
            AgentSystem.instance().send_message(response)

class AgentSystem:
    """Singleton class that manages all agents and message passing"""
    
    _instance = None
    
    @classmethod
    def instance(cls):
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = AgentSystem()
        return cls._instance
    
    def __init__(self):
        self.agents = {}  # id -> agent
        self.topics = {}  # topic -> set of subscriber ids
        self.running = False
        self.message_count = 0
    
    def register_agent(self, agent: Agent):
        """Register an agent with the system"""
        self.agents[agent.id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.id})")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            # Unsubscribe from all topics
            for topic in list(self.topics.keys()):
                self.unsubscribe(agent_id, topic)
            
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def start_all_agents(self):
        """Start all registered agents"""
        self.running = True
        for agent in self.agents.values():
            agent.start()
        logger.info(f"Started all agents ({len(self.agents)})")
    
    def stop_all_agents(self):
        """Stop all registered agents"""
        self.running = False
        for agent in self.agents.values():
            agent.stop()
        logger.info(f"Stopped all agents ({len(self.agents)})")
    
    def send_message(self, message: Message):
        """Send a message to an agent"""
        if not self.running:
            logger.warning(f"Cannot send message - agent system is not running")
            return False
        
        self.message_count += 1
        
        if message.receiver in self.agents:
            self.agents[message.receiver].receive_message(message)
            logger.debug(f"Sent message: {message}")
            return True
        else:
            logger.warning(f"Cannot send message to unknown agent: {message.receiver}")
            return False
    
    def subscribe(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic"""
        if topic not in self.topics:
            self.topics[topic] = set()
        
        self.topics[topic].add(agent_id)
        logger.debug(f"Agent {agent_id} subscribed to topic {topic}")
    
    def unsubscribe(self, agent_id: str, topic: str):
        """Unsubscribe an agent from a topic"""
        if topic in self.topics and agent_id in self.topics[topic]:
            self.topics[topic].remove(agent_id)
            
            # Remove topic if there are no more subscribers
            if not self.topics[topic]:
                del self.topics[topic]
            
            logger.debug(f"Agent {agent_id} unsubscribed from topic {topic}")
    
    def publish(self, sender_id: str, topic: str, content: Any, metadata: Dict[str, Any] = None) -> Message:
        """Publish content to a topic"""
        if not self.running:
            logger.warning(f"Cannot publish message - agent system is not running")
            return None
        
        if topic not in self.topics or not self.topics[topic]:
            logger.debug(f"No subscribers for topic {topic}")
            return None
        
        sent_count = 0
        message = None
        
        # Create a copy of the set to avoid modification during iteration
        subscribers = self.topics[topic].copy()
        
        # Send to each subscriber
        for agent_id in subscribers:
            if agent_id in self.agents:
                message = Message(
                    sender=sender_id,
                    receiver=agent_id,
                    message_type=MessageType.EVENT,
                    content=content,
                    metadata=metadata or {"topic": topic}
                )
                
                if "topic" not in message.metadata:
                    message.metadata["topic"] = topic
                
                self.send_message(message)
                sent_count += 1
        
        logger.debug(f"Published message to topic {topic} ({sent_count} recipients)")
        return message
    
    def broadcast(self, sender_id: str, message_type: MessageType, content: Any, 
                 metadata: Dict[str, Any] = None) -> int:
        """Broadcast a message to all agents"""
        if not self.running:
            logger.warning(f"Cannot broadcast message - agent system is not running")
            return 0
        
        sent_count = 0
        
        for agent_id in self.agents:
            if agent_id != sender_id:  # Don't send to self
                message = Message(
                    sender=sender_id,
                    receiver=agent_id,
                    message_type=message_type,
                    content=content,
                    metadata=metadata
                )
                
                if self.send_message(message):
                    sent_count += 1
        
        logger.debug(f"Broadcast message to {sent_count} agents")
        return sent_count

class VigilanceAgent(Agent):
    """An agent dedicated to monitoring and vigilance"""
    
    def __init__(self, agent_id: str, name: str = "Vigilance Agent"):
        super().__init__(agent_id, name)
        
        # Register message handlers
        self.register_handler(MessageType.EVENT, self._handle_event)
        self.register_handler(MessageType.REQUEST, self._handle_request)
        
        # Alert thresholds
        self.alert_level = 0  # 0-100
        self.alert_threshold = 70  # Threshold for sending alerts
        
        # Vigilance settings
        self.threat_patterns = {}  # pattern_id -> pattern_def
        self.recent_events = []
        self.max_events = 1000
    
    def add_threat_pattern(self, pattern_id: str, pattern_def: Dict[str, Any]):
        """Add a threat pattern to monitor for"""
        self.threat_patterns[pattern_id] = pattern_def
        logger.info(f"Added threat pattern {pattern_id} to vigilance agent")
    
    def remove_threat_pattern(self, pattern_id: str):
        """Remove a threat pattern"""
        if pattern_id in self.threat_patterns:
            del self.threat_patterns[pattern_id]
            logger.info(f"Removed threat pattern {pattern_id} from vigilance agent")
    
    def set_alert_threshold(self, threshold: int):
        """Set the alert threshold (0-100)"""
        self.alert_threshold = max(0, min(100, threshold))
    
    def _handle_event(self, message: Message):
        """Handle an event message"""
        # Add to recent events
        self.recent_events.append({
            "timestamp": message.timestamp,
            "content": message.content,
            "metadata": message.metadata,
            "source": message.sender
        })
        
        # Keep only the most recent events
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events:]
        
        # Check for threat patterns
        self._check_threat_patterns()
    
    def _handle_request(self, message: Message):
        """Handle a request message"""
        request_type = message.content.get("type")
        
        if request_type == "status":
            # Return agent status
            response = message.create_response({
                "status": "active",
                "alert_level": self.alert_level,
                "alert_threshold": self.alert_threshold,
                "patterns": len(self.threat_patterns),
                "recent_events": len(self.recent_events)
            })
            AgentSystem.instance().send_message(response)
        
        elif request_type == "set_threshold":
            # Set alert threshold
            threshold = message.content.get("threshold")
            if threshold is not None:
                self.set_alert_threshold(threshold)
                response = message.create_response({
                    "status": "success",
                    "new_threshold": self.alert_threshold
                })
            else:
                response = message.create_response({
                    "status": "error",
                    "message": "Missing threshold parameter"
                })
            
            AgentSystem.instance().send_message(response)
        
        else:
            # Unknown request type
            response = message.create_response({
                "status": "error",
                "message": f"Unknown request type: {request_type}"
            })
            AgentSystem.instance().send_message(response)
    
    def _check_threat_patterns(self):
        """Check recent events against threat patterns"""
        # This is a simplified pattern matching implementation
        # A real implementation would be more sophisticated
        
        for pattern_id, pattern in self.threat_patterns.items():
            matches = []
            confidence = 0.0
            
            # Check if recent events match the pattern
            for event in reversed(self.recent_events[-100:]):  # Check last 100 events
                if self._event_matches_pattern(event, pattern):
                    matches.append(event)
                    confidence += 0.1  # Increase confidence with each match
            
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            # If confidence is high enough, send an alert
            alert_level = int(confidence * 100)
            if matches and alert_level > self.alert_threshold:
                self._send_alert(pattern_id, pattern, matches, alert_level)
    
    def _event_matches_pattern(self, event: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if an event matches a pattern"""
        # Check for required fields
        if "indicators" in pattern:
            for indicator, value in pattern["indicators"].items():
                if indicator not in event["content"] or event["content"][indicator] != value:
                    return False
        
        # Check for source constraints
        if "sources" in pattern and event["source"] not in pattern["sources"]:
            return False
        
        return True
    
    def _send_alert(self, pattern_id: str, pattern: Dict[str, Any], matches: List[Dict[str, Any]], alert_level: int):
        """Send an alert about a detected threat pattern"""
        # Update our alert level
        self.alert_level = max(self.alert_level, alert_level)
        
        # Create alert content
        alert_content = {
            "pattern_id": pattern_id,
            "pattern_name": pattern.get("name", pattern_id),
            "description": pattern.get("description", "No description"),
            "alert_level": alert_level,
            "match_count": len(matches),
            "timestamp": time.time()
        }
        
        # Publish alert to the alerts topic
        self.publish("alerts", alert_content)
        
        logger.warning(f"Alert: {pattern.get('name', pattern_id)} (level: {alert_level})")

class DecisionAgent(Agent):
    """An agent dedicated to making decisions based on alerts and context"""
    
    def __init__(self, agent_id: str, name: str = "Decision Agent"):
        super().__init__(agent_id, name)
        
        # Register message handlers
        self.register_handler(MessageType.EVENT, self._handle_event)
        self.register_handler(MessageType.REQUEST, self._handle_request)
        
        # Subscribe to alerts
        self.subscribe("alerts")
        
        # Decision settings
        self.action_templates = {}  # template_id -> template
        self.context = {}  # Current context
        self.active_alerts = []  # List of active alerts
        self.max_alerts = 100
        self.alert_timeout = 3600  # Seconds until an alert is considered stale
    
    def add_action_template(self, template_id: str, template: Dict[str, Any]):
        """Add an action template"""
        self.action_templates[template_id] = template
        logger.info(f"Added action template {template_id} to decision agent")
    
    def remove_action_template(self, template_id: str):
        """Remove an action template"""
        if template_id in self.action_templates:
            del self.action_templates[template_id]
            logger.info(f"Removed action template {template_id} from decision agent")
    
    def update_context(self, key: str, value: Any):
        """Update the current context"""
        self.context[key] = value
        logger.debug(f"Updated context: {key} = {value}")
    
    def _handle_event(self, message: Message):
        """Handle an event message"""
        # Check if this is an alert
        if message.metadata.get("topic") == "alerts":
            self._handle_alert(message.content)
        else:
            # Update context based on event
            self._update_context_from_event(message)
    
    def _handle_request(self, message: Message):
        """Handle a request message"""
        request_type = message.content.get("type")
        
        if request_type == "status":
            # Return agent status
            response = message.create_response({
                "status": "active",
                "action_templates": len(self.action_templates),
                "active_alerts": len(self.active_alerts),
                "context_keys": list(self.context.keys())
            })
            AgentSystem.instance().send_message(response)
        
        elif request_type == "get_recommendations":
            # Generate action recommendations
            actions = self._generate_actions()
            response = message.create_response({
                "status": "success",
                "actions": actions
            })
            AgentSystem.instance().send_message(response)
        
        else:
            # Unknown request type
            response = message.create_response({
                "status": "error",
                "message": f"Unknown request type: {request_type}"
            })
            AgentSystem.instance().send_message(response)
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """Handle an alert"""
        # Add to active alerts
        self.active_alerts.append(alert)
        
        # Keep only the most recent alerts
        if len(self.active_alerts) > self.max_alerts:
            self.active_alerts = self.active_alerts[-self.max_alerts:]
        
        # Remove stale alerts
        current_time = time.time()
        self.active_alerts = [a for a in self.active_alerts 
                              if current_time - a["timestamp"] <= self.alert_timeout]
        
        # Generate actions based on the new alert
        actions = self._generate_actions()
        
        # Publish recommended actions
        if actions:
            self.publish("recommendations", {
                "actions": actions,
                "trigger_alert": alert,
                "timestamp": time.time()
            })
    
    def _update_context_from_event(self, message: Message):
        """Update context based on an event"""
        # This is a simplified implementation
        # A real implementation would have more sophisticated context updating
        
        # For example, extract some fields from the message
        if isinstance(message.content, dict):
            for key, value in message.content.items():
                context_key = f"{message.sender}.{key}"
                self.update_context(context_key, value)
    
    def _generate_actions(self) -> List[Dict[str, Any]]:
        """Generate recommended actions based on alerts and context"""
        if not self.active_alerts:
            return []
        
        recommended_actions = []
        
        for alert in self.active_alerts:
            # Find applicable action templates
            applicable_templates = self._find_applicable_templates(alert)
            
            for template_id, template in applicable_templates.items():
                # Calculate effectiveness
                effectiveness = template.get("base_effectiveness", 0.5)
                if "effectiveness_factors" in template:
                    for factor, weight in template["effectiveness_factors"].items():
                        if factor == "alert_level":
                            effectiveness += (alert["alert_level"] / 100) * weight
                        elif factor in self.context:
                            effectiveness += float(self.context[factor]) * weight
                
                effectiveness = max(0.0, min(1.0, effectiveness))
                
                # Calculate resource cost
                resource_cost = template.get("base_resource_cost", 0.5)
                if "cost_factors" in template:
                    for factor, weight in template["cost_factors"].items():
                        if factor in self.context:
                            resource_cost += float(self.context[factor]) * weight
                
                resource_cost = max(0.1, min(1.0, resource_cost))
                
                # Only recommend if effective enough
                if effectiveness > 0.3:
                    action = {
                        "id": template_id,
                        "name": template.get("name", template_id),
                        "description": template.get("description", "No description"),
                        "effectiveness": effectiveness,
                        "resource_cost": resource_cost,
                        "trigger_alert": {
                            "pattern_id": alert["pattern_id"],
                            "level": alert["alert_level"]
                        },
                        "timestamp": time.time()
                    }
                    
                    recommended_actions.append(action)
        
        # Sort by effectiveness
        recommended_actions.sort(key=lambda x: x["effectiveness"], reverse=True)
        
        return recommended_actions
    
    def _find_applicable_templates(self, alert: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Find action templates applicable to an alert"""
        applicable = {}
        
        for template_id, template in self.action_templates.items():
            # Check if template applies to this alert
            if "applicable_alerts" in template:
                if alert["pattern_id"] not in template["applicable_alerts"]:
                    continue
            
            # Check if prerequisites are met
            if "prerequisites" in template:
                prereqs_met = True
                for prereq, value in template["prerequisites"].items():
                    if prereq not in self.context or self.context[prereq] != value:
                        prereqs_met = False
                        break
                
                if not prereqs_met:
                    continue
            
            applicable[template_id] = template
        
        return applicable

# Example usage (for demonstration)
if __name__ == "__main__":
    # Initialize agent system
    system = AgentSystem.instance()
    
    # Create a vigilance agent
    vigilance = VigilanceAgent("vigilance1", "Primary Vigilance Agent")
    
    # Add some threat patterns
    vigilance.add_threat_pattern(
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
    
    # Create a decision agent
    decision = DecisionAgent("decision1", "Primary Decision Agent")
    
    # Add some action templates
    decision.add_action_template(
        "lockdown_systems",
        {
            "name": "Lock Down Systems",
            "description": "Temporarily restrict all access to critical systems",
            "applicable_alerts": ["unauthorized_access"],
            "base_effectiveness": 0.8,
            "base_resource_cost": 0.7
        }
    )
    
    # Register agents
    system.register_agent(vigilance)
    system.register_agent(decision)
    
    # Start all agents
    system.start_all_agents()
    
    # Simulate an event
    vigilance.receive_message(Message(
        sender="security_sensor",
        receiver="vigilance1",
        message_type=MessageType.EVENT,
        content={
            "access_attempt": True,
            "authorized": False,
            "resource": "admin_panel",
            "ip_address": "198.51.100.1"
        }
    ))
    
    # Give some time for processing
    time.sleep(1)
    
    # Stop all agents
    system.stop_all_agents()