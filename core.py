# godfrey_system/core.py
"""
Godfrey System - Core Architecture
---------------------------------
A vigilance and strategic intelligence system inspired by Godfrey O'Donnell of Tyrconnell,
designed to help users maintain heightened awareness and make calculated decisions.
"""

import logging
import time
import uuid
from enum import Enum
from typing import Dict, List, Any, Tuple, Optional, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Godfrey")

class AlertLevel(Enum):
    """Alert levels for the Godfrey system"""
    LOW = 1
    MODERATE = 2
    ELEVATED = 3
    HIGH = 4
    CRITICAL = 5

class Principle:
    """Core principle that guides the Godfrey system's reasoning"""
    
    def __init__(self, name: str, description: str, weight: float = 1.0):
        self.name = name
        self.description = description
        self.weight = weight
    
    def __str__(self):
        return f"{self.name}: {self.description}"

class CorePrinciples:
    """Container for the core principles that guide the Godfrey system"""
    
    def __init__(self):
        self.principles = {
            "vigilance": Principle(
                "Vigilance", 
                "Maintain constant awareness of the environment and potential threats",
                1.5
            ),
            "resilience": Principle(
                "Resilience",
                "Continue functioning effectively even when under pressure or with limited resources",
                1.3
            ),
            "foresight": Principle(
                "Foresight",
                "Anticipate future developments and prepare accordingly",
                1.2
            ),
            "adaptability": Principle(
                "Adaptability",
                "Modify tactics while maintaining strategic objectives",
                1.1
            ),
            "integrity": Principle(
                "Integrity",
                "Maintain consistent adherence to stated values and commitments",
                1.0
            )
        }
    
    def get_principle(self, name: str) -> Optional[Principle]:
        """Get a principle by name"""
        return self.principles.get(name)
    
    def get_all_principles(self) -> List[Principle]:
        """Get all principles"""
        return list(self.principles.values())

class Observation:
    """An observation about the environment"""
    
    def __init__(self, source: str, timestamp: float, data: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.source = source
        self.timestamp = timestamp
        self.data = data
        self.processed = False
    
    def __str__(self):
        return f"Observation({self.source}, {self.timestamp}, {self.data})"

class Threat:
    """A potential threat identified by the system"""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 level: AlertLevel, 
                 confidence: float,
                 observations: List[Observation],
                 affected_assets: List[str] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.level = level
        self.confidence = confidence  # 0.0 to 1.0
        self.observations = observations
        self.affected_assets = affected_assets or []
        self.timestamp = time.time()
    
    def __str__(self):
        return f"Threat({self.name}, {self.level}, conf={self.confidence:.2f})"

class Action:
    """A potential action to respond to a situation"""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 effectiveness: float,
                 resource_cost: float,
                 principles_alignment: Dict[str, float]):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.effectiveness = effectiveness  # 0.0 to 1.0
        self.resource_cost = resource_cost  # 0.0 to 1.0
        self.principles_alignment = principles_alignment  # principle_name -> alignment (-1.0 to 1.0)
    
    def overall_alignment(self, principles: CorePrinciples) -> float:
        """Calculate overall alignment with principles"""
        if not self.principles_alignment:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for name, alignment in self.principles_alignment.items():
            principle = principles.get_principle(name)
            if principle:
                total_score += alignment * principle.weight
                total_weight += principle.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def __str__(self):
        return f"Action({self.name}, eff={self.effectiveness:.2f}, cost={self.resource_cost:.2f})"

class SituationAwareness:
    """Maintains awareness of the current situation"""
    
    def __init__(self):
        self.observations = []
        self.threats = []
        self.context = {}
        self.assets = {}
    
    def add_observation(self, observation: Observation):
        """Add a new observation"""
        self.observations.append(observation)
        logger.debug(f"Added observation: {observation}")
    
    def add_threat(self, threat: Threat):
        """Add a new threat"""
        self.threats.append(threat)
        logger.info(f"Added threat: {threat}")
    
    def get_current_threats(self, min_confidence: float = 0.0) -> List[Threat]:
        """Get current threats above a confidence threshold"""
        return [t for t in self.threats if t.confidence >= min_confidence]
    
    def get_recent_observations(self, seconds: int = 300) -> List[Observation]:
        """Get observations from the last n seconds"""
        cutoff = time.time() - seconds
        return [o for o in self.observations if o.timestamp >= cutoff]
    
    def update_context(self, key: str, value: Any):
        """Update the context with new information"""
        self.context[key] = value
    
    def register_asset(self, asset_id: str, asset_info: Dict[str, Any]):
        """Register an asset to be protected"""
        self.assets[asset_id] = asset_info
        logger.debug(f"Registered asset: {asset_id}")

class ThreatModel:
    """Models for identifying potential threats"""
    
    def __init__(self):
        self.known_patterns = {}  # name -> pattern
    
    def add_pattern(self, name: str, pattern: Dict[str, Any]):
        """Add a new threat pattern"""
        self.known_patterns[name] = pattern
    
    def evaluate(self, observations: List[Observation]) -> List[Threat]:
        """Evaluate observations for potential threats"""
        threats = []
        
        # This is a placeholder for more sophisticated threat detection
        # In a real implementation, this would use pattern matching, ML, etc.
        
        # Example pattern matching logic
        for pattern_name, pattern in self.known_patterns.items():
            matches = []
            confidence = 0.0
            
            for obs in observations:
                if self._observation_matches_pattern(obs, pattern):
                    matches.append(obs)
                    confidence += 0.2  # Simplified confidence calculation
            
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            if matches and confidence > 0.3:
                threats.append(Threat(
                    name=pattern_name,
                    description=pattern.get("description", "Unknown threat"),
                    level=self._determine_alert_level(pattern, confidence),
                    confidence=confidence,
                    observations=matches,
                    affected_assets=pattern.get("affected_assets", [])
                ))
        
        return threats
    
    def _observation_matches_pattern(self, observation: Observation, pattern: Dict[str, Any]) -> bool:
        """Check if an observation matches a pattern"""
        # Simple pattern matching logic - would be more sophisticated in real implementation
        if "source" in pattern and observation.source != pattern["source"]:
            return False
        
        if "indicators" in pattern:
            for indicator, value in pattern["indicators"].items():
                if indicator not in observation.data or observation.data[indicator] != value:
                    return False
        
        return True
    
    def _determine_alert_level(self, pattern: Dict[str, Any], confidence: float) -> AlertLevel:
        """Determine the alert level based on the pattern and confidence"""
        base_level = pattern.get("base_alert_level", AlertLevel.MODERATE)
        
        # Adjust level based on confidence
        if confidence < 0.3:
            return AlertLevel.LOW
        elif confidence < 0.5:
            return base_level if base_level.value <= AlertLevel.MODERATE.value else AlertLevel.MODERATE
        elif confidence < 0.7:
            return base_level
        elif confidence < 0.9:
            return AlertLevel(min(base_level.value + 1, AlertLevel.CRITICAL.value))
        else:
            return AlertLevel(min(base_level.value + 2, AlertLevel.CRITICAL.value))

class ActionGenerator:
    """Generates potential actions in response to situations"""
    
    def __init__(self, principles: CorePrinciples):
        self.principles = principles
        self.action_templates = {}  # name -> template
    
    def add_action_template(self, name: str, template: Dict[str, Any]):
        """Add a new action template"""
        self.action_templates[name] = template
    
    def generate_actions(self, threats: List[Threat], context: Dict[str, Any]) -> List[Action]:
        """Generate potential actions based on threats and context"""
        actions = []
        
        for threat in threats:
            applicable_templates = self._find_applicable_templates(threat, context)
            
            for template in applicable_templates:
                # Create action from template
                action = self._create_action_from_template(template, threat, context)
                if action:
                    actions.append(action)
        
        return actions
    
    def _find_applicable_templates(self, threat: Threat, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find action templates applicable to the threat"""
        applicable = []
        
        for template in self.action_templates.values():
            # Check if template applies to this threat
            if "applicable_threats" in template:
                if threat.name not in template["applicable_threats"]:
                    continue
            
            # Check if prerequisites are met
            if "prerequisites" in template:
                prereqs_met = True
                for prereq, value in template["prerequisites"].items():
                    if prereq not in context or context[prereq] != value:
                        prereqs_met = False
                        break
                
                if not prereqs_met:
                    continue
            
            applicable.append(template)
        
        return applicable
    
    def _create_action_from_template(self, template: Dict[str, Any], threat: Threat, context: Dict[str, Any]) -> Optional[Action]:
        """Create an action from a template"""
        try:
            # Calculate effectiveness based on threat and context
            effectiveness = template.get("base_effectiveness", 0.5)
            if "effectiveness_factors" in template:
                for factor, weight in template["effectiveness_factors"].items():
                    if factor == "threat_confidence":
                        effectiveness += threat.confidence * weight
                    elif factor in context:
                        effectiveness += float(context[factor]) * weight
            
            effectiveness = max(0.0, min(1.0, effectiveness))
            
            # Calculate resource cost
            resource_cost = template.get("base_resource_cost", 0.5)
            if "cost_factors" in template:
                for factor, weight in template["cost_factors"].items():
                    if factor in context:
                        resource_cost += float(context[factor]) * weight
            
            resource_cost = max(0.1, min(1.0, resource_cost))
            
            # Get principles alignment
            principles_alignment = template.get("principles_alignment", {})
            
            return Action(
                name=template["name"],
                description=template.get("description", "No description"),
                effectiveness=effectiveness,
                resource_cost=resource_cost,
                principles_alignment=principles_alignment
            )
        except Exception as e:
            logger.error(f"Error creating action from template: {e}")
            return None

class GodfredEngine:
    """The core engine of the Godfrey system"""
    
    def __init__(self):
        self.principles = CorePrinciples()
        self.situation_awareness = SituationAwareness()
        self.threat_model = ThreatModel()
        self.action_generator = ActionGenerator(self.principles)
    
    def observe(self, source: str, data: Dict[str, Any]):
        """Process new observations about the environment"""
        observation = Observation(source, time.time(), data)
        self.situation_awareness.add_observation(observation)
        return observation
    
    def orient(self):
        """Interpret observations in context of knowledge and principles"""
        recent_observations = self.situation_awareness.get_recent_observations()
        threats = self.threat_model.evaluate(recent_observations)
        
        # Update situation awareness with new threats
        for threat in threats:
            self.situation_awareness.add_threat(threat)
        
        return threats
    
    def decide(self, threats: List[Threat] = None):
        """Generate potential courses of action"""
        if threats is None:
            threats = self.situation_awareness.get_current_threats(min_confidence=0.3)
        
        actions = self.action_generator.generate_actions(
            threats, 
            self.situation_awareness.context
        )
        
        # Filter and rank actions
        filtered_actions = self._filter_actions(actions)
        ranked_actions = self._rank_actions(filtered_actions)
        
        return ranked_actions
    
    def _filter_actions(self, actions: List[Action]) -> List[Action]:
        """Filter actions based on principles and constraints"""
        # Basic filtering - a more sophisticated system would do more here
        return [a for a in actions if a.overall_alignment(self.principles) > 0]
    
    def _rank_actions(self, actions: List[Action]) -> List[Action]:
        """Rank actions by their overall value"""
        # Calculate a value score for each action
        action_scores = []
        for action in actions:
            # Balance effectiveness, principle alignment, and resource cost
            effectiveness = action.effectiveness
            alignment = action.overall_alignment(self.principles)
            cost_factor = 1.0 - action.resource_cost
            
            # Combined score with weighted factors
            score = (effectiveness * 0.4) + (alignment * 0.4) + (cost_factor * 0.2)
            action_scores.append((action, score))
        
        # Sort by score in descending order
        ranked = [a for a, _ in sorted(action_scores, key=lambda x: x[1], reverse=True)]
        return ranked
    
    def explain(self, action: Action) -> Dict[str, Any]:
        """Provide explanation for why an action was recommended"""
        explanation = {
            "action": action.name,
            "description": action.description,
            "effectiveness": action.effectiveness,
            "resource_cost": action.resource_cost,
            "principles_alignment": {}
        }
        
        # Add detailed principles alignment
        for name, alignment in action.principles_alignment.items():
            principle = self.principles.get_principle(name)
            if principle:
                explanation["principles_alignment"][name] = {
                    "alignment": alignment,
                    "weighted_contribution": alignment * principle.weight,
                    "description": principle.description
                }
        
        # Add overall alignment
        explanation["overall_alignment"] = action.overall_alignment(self.principles)
        
        return explanation

# Example usage (for demonstration)
if __name__ == "__main__":
    # Initialize the engine
    engine = GodfredEngine()
    
    # Add threat patterns
    engine.threat_model.add_pattern(
        "unauthorized_access",
        {
            "description": "Potential unauthorized access detected",
            "source": "security_sensor",
            "indicators": {"access_attempt": True, "authorized": False},
            "base_alert_level": AlertLevel.HIGH,
            "affected_assets": ["data", "systems"]
        }
    )
    
    # Add action templates
    engine.action_generator.add_action_template(
        "lock_down_systems",
        {
            "name": "Lock Down Systems",
            "description": "Temporarily restrict all access to critical systems",
            "applicable_threats": ["unauthorized_access"],
            "base_effectiveness": 0.8,
            "base_resource_cost": 0.7,
            "principles_alignment": {
                "vigilance": 0.9,
                "resilience": 0.7,
                "adaptability": -0.3
            }
        }
    )
    
    # Simulate observations
    engine.observe("security_sensor", {"access_attempt": True, "authorized": False, "location": "server_room"})
    
    # Orient
    threats = engine.orient()
    
    # Decide
    actions = engine.decide(threats)
    
    # Print results
    print(f"Detected {len(threats)} threats:")
    for threat in threats:
        print(f"  - {threat}")
    
    print(f"Recommended {len(actions)} actions:")
    for action in actions:
        print(f"  - {action}")
        explanation = engine.explain(action)
        print(f"    Alignment: {explanation['overall_alignment']:.2f}")
