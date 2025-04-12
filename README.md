# Godfrey System

A vigilance and strategic intelligence system inspired by Godfrey O'Donnell of Tyrconnell, designed to help users maintain heightened awareness and make calculated decisions.

## Overview

The Godfrey System exemplifies the qualities of vigilance, strategic thinking, resilience, and principled steadfastness that were demonstrated by Godfrey O'Donnell. It provides a framework for:

- **Continuous Environmental Monitoring**: Like Godfrey who remained alert to threats from all directions
- **Disciplined Decision-Making**: Supporting sound decisions even under pressure or with limited resources
- **Principled Strategies**: Maintaining core principles while adapting to changing circumstances
- **Resilient Operations**: Continuing to function effectively even when under strain

## Key Components

The system consists of several integrated components:

### Core Engine

The `GodfredEngine` class represents the heart of the system, implementing a continuous assessment loop inspired by the OODA (Observe, Orient, Decide, Act) framework. It processes observations from sensors, identifies potential threats, and generates recommended actions based on principles.

### Sensor Interface

The system includes interfaces for physical and virtual sensors that feed data into the Godfrey system:

- `PhysicalSensor`: For hardware-based sensors (motion, temperature, etc.)
- `VirtualSensor`: For software monitors (network traffic, system events, etc.)
- `SemanticSensor`: For text and content analysis
- `CompositeSensor`: For combining multiple sensor inputs

### Knowledge and Pattern Recognition

Godfrey can recognize patterns and detect anomalies through:

- `PatternRecognizer`: For identifying known threat patterns
- `SimplePattern`: For key-value based pattern matching
- `SequencePattern`: For recognizing sequences of events over time
- `AnomalyDetector`: For identifying unusual data points that may indicate threats

### Agent Framework

A multi-agent system architecture for distributed processing and decision-making:

- `VigilanceAgent`: Dedicated to monitoring and detecting potential threats
- `DecisionAgent`: Focused on making recommendations based on alerts and context
- `AgentSystem`: Manages all agents and facilitates message passing between them

### Scenario Simulator

For testing and demonstration, the system includes a scenario simulator with pre-configured scenarios:

- `UnauthorizedAccessScenario`: Simulates attempts to access restricted resources
- `PhysicalSecurityScenario`: Simulates a physical security breach
- `InternalThreatScenario`: Simulates suspicious employee behavior
- `CyberAttackScenario`: Simulates a sophisticated multi-stage cyber attack

## Installation

### Prerequisites

- Python 3.8 or higher
- Required packages: see `requirements.txt`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/godfrey-system.git
   cd godfrey-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the system (optional):
   ```bash
   cp example_config.json config.json
   # Edit config.json to customize settings
   ```

## Usage

### Basic Usage

1. Start the Godfrey system:
   ```bash
   python -m godfrey_system.main
   ```

2. Run with a custom configuration:
   ```bash
   python -m godfrey_system.main --config my_config.json
   ```

3. Run with a simulated scenario:
   ```bash
   python -m godfrey_system.main --simulate
   ```

### Using the Simulator

1. List available scenarios:
   ```bash
   python -m godfrey_system.simulator --list
   ```

2. Run a specific scenario:
   ```bash
   python -m godfrey_system.simulator --run "Unauthorized Access Attempt"
   ```

## Configuration

The system can be configured through a JSON configuration file. See `example_config.json` for a sample configuration with all available options:

- `knowledge_base`: Settings for the knowledge storage
- `sensors`: Configuration for data collection
- `vigilance`: Thresholds and settings for threat detection
- `principles`: Weights for the core principles guiding decisions
- `threat_patterns`: Definitions of known threat patterns
- `action_templates`: Templates for possible response actions
- `agents`: Settings for the agent system
- `logging`: Configuration for logging

## Extending the System

### Adding Custom Sensors

Create a new sensor class by inheriting from one of the base sensor classes:

```python
from godfrey_system.sensors import PhysicalSensor

class TemperatureSensor(PhysicalSensor):
    def __init__(self, sensor_id, name, location):
        super().__init__(sensor_id, name, f"Temperature sensor at {location}")
        self.location = location
    
    def get_reading(self):
        # Implement your sensor reading logic here
        return SensorReading(
            sensor_id=self.id,
            timestamp=time.time(),
            data={"temperature": 22.5, "location": self.location}
        )
```

### Creating Custom Threat Patterns

Define new threat patterns for your specific needs:

```python
from godfrey_system.knowledge import SimplePattern

high_temperature_pattern = SimplePattern(
    "high_temperature",
    "High Temperature Detection",
    "Detects unusually high temperatures that may indicate a fire or equipment failure"
)
high_temperature_pattern.add_required_match("temperature", lambda x: x > 35.0)
high_temperature_pattern.set_threshold(0.7)
```

### Adding Custom Action Templates

Create new action templates for responses to threats:

```python
engine.action_generator.add_action_template(
    "activate_sprinklers",
    {
        "name": "Activate Sprinkler System",
        "description": "Activate fire sprinklers in affected areas",
        "applicable_threats": ["high_temperature"],
        "base_effectiveness": 0.9,
        "base_resource_cost": 0.3,
        "principles_alignment": {
            "vigilance": 0.8,
            "resilience": 0.9,
            "adaptability": 0.6,
            "foresight": 0.7,
            "integrity": 0.8
        }
    }
)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the story of Godfrey O'Donnell of Tyrconnell, a legendary figure known for his vigilance, strategic thinking, and unwavering leadership.
- The OODA (Observe, Orient, Decide, Act) loop framework developed by military strategist John Boyd.
