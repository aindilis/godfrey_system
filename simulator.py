# godfrey_system/simulator.py
"""
Godfrey System - Scenario Simulator
----------------------------------
Simulates various scenarios to test and demonstrate the Godfrey system capabilities.
"""

import time
import random
import threading
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from godfrey_system.agents import Message, MessageType, AgentSystem
from godfrey_system.core import AlertLevel

logger = logging.getLogger("Godfrey.Simulator")

class SimulationScenario:
    """Base class for simulation scenarios"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.events = []  # List of (time_offset, event_fn) tuples
        self.running = False
        self.start_time = None
    
    def add_event(self, time_offset: float, event_fn):
        """Add an event to the scenario at the specified time offset (in seconds)"""
        self.events.append((time_offset, event_fn))
        # Sort events by time offset
        self.events.sort(key=lambda x: x[0])
    
    def run(self, system):
        """Run the scenario"""
        if self.running:
            logger.warning(f"Scenario '{self.name}' is already running")
            return
        
        logger.info(f"Starting scenario: {self.name}")
        logger.info(f"Description: {self.description}")
        
        self.running = True
        self.start_time = time.time()
        
        # Run in a separate thread
        thread = threading.Thread(target=self._run_scenario, args=(system,))
        thread.daemon = True
        thread.start()
        
        return thread
    
    def _run_scenario(self, system):
        """Run the scenario events"""
        try:
            for time_offset, event_fn in self.events:
                # Calculate time to wait
                now = time.time()
                elapsed = now - self.start_time
                wait_time = max(0, time_offset - elapsed)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # Run the event if we're still running
                if self.running:
                    try:
                        event_fn(system)
                    except Exception as e:
                        logger.error(f"Error in scenario event: {e}")
            
            logger.info(f"Scenario '{self.name}' completed")
        
        except Exception as e:
            logger.error(f"Error running scenario '{self.name}': {e}")
        
        finally:
            self.running = False
    
    def stop(self):
        """Stop the scenario"""
        if not self.running:
            return
        
        logger.info(f"Stopping scenario: {self.name}")
        self.running = False

class UnauthorizedAccessScenario(SimulationScenario):
    """Simulates an unauthorized access attempt scenario"""
    
    def __init__(self):
        super().__init__(
            "Unauthorized Access Attempt",
            "Simulates a scenario where an unauthorized user attempts to access secure systems"
        )
        
        # Add events to the scenario
        self.add_event(0, self._initial_network_scan)
        self.add_event(10, self._failed_login_attempt)
        self.add_event(15, self._another_failed_login)
        self.add_event(30, self._suspicious_network_traffic)
        self.add_event(45, self._successful_login)
        self.add_event(60, self._access_sensitive_data)
    
    def _initial_network_scan(self, system):
        """Simulate initial network scanning"""
        logger.info("Event: Initial network scanning detected")
        
        network_sensor = system.sensor_manager.get_sensor("network1")
        if network_sensor:
            network_sensor.receive_message(Message(
                sender="simulation",
                receiver=network_sensor.id,
                message_type=MessageType.EVENT,
                content={
                    "traffic_level": 0.4,
                    "suspicious_activity": True,
                    "anomaly_score": 0.3,
                    "alert": False
                }
            ))
    
    def _failed_login_attempt(self, system):
        """Simulate a failed login attempt"""
        logger.info("Event: Failed login attempt")
        
        # Simulate through a custom event
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="auth_system",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "access_attempt": True,
                    "authorized": False,
                    "user": "unknown",
                    "resource": "admin_panel",
                    "ip_address": "198.51.100.1"
                }
            ))
    
    def _another_failed_login(self, system):
        """Simulate another failed login attempt"""
        logger.info("Event: Another failed login attempt")
        
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="auth_system",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "access_attempt": True,
                    "authorized": False,
                    "user": "unknown",
                    "resource": "admin_panel",
                    "ip_address": "198.51.100.1"
                }
            ))
    
    def _suspicious_network_traffic(self, system):
        """Simulate suspicious network traffic"""
        logger.info("Event: Suspicious network traffic detected")
        
        network_sensor = system.sensor_manager.get_sensor("network1")
        if network_sensor:
            network_sensor.receive_message(Message(
                sender="simulation",
                receiver=network_sensor.id,
                message_type=MessageType.EVENT,
                content={
                    "traffic_level": 0.7,
                    "suspicious_activity": True,
                    "unknown_ip": "203.0.113.100",
                    "anomaly_score": 0.75,
                    "alert": True
                }
            ))
    
    def _successful_login(self, system):
        """Simulate a successful but suspicious login"""
        logger.info("Event: Successful but suspicious login")
        
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="auth_system",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "access_attempt": True,
                    "authorized": True,  # Successful login
                    "user": "admin",
                    "resource": "admin_panel",
                    "ip_address": "198.51.100.1",  # Same suspicious IP
                    "unusual_time": True,
                    "unusual_location": True
                }
            ))
    
    def _access_sensitive_data(self, system):
        """Simulate access to sensitive data"""
        logger.info("Event: Accessing sensitive data")
        
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="file_system",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "file_access": True,
                    "file_path": "/sensitive/user_data.db",
                    "user": "admin",
                    "access_type": "read",
                    "unusual_access_pattern": True
                }
            ))

class PhysicalSecurityScenario(SimulationScenario):
    """Simulates a physical security breach scenario"""
    
    def __init__(self):
        super().__init__(
            "Physical Security Breach",
            "Simulates a scenario where an unauthorized person breaches physical security"
        )
        
        # Add events to the scenario
        self.add_event(0, self._perimeter_motion)
        self.add_event(10, self._door_sensor_trigger)
        self.add_event(20, self._camera_motion_detection)
        self.add_event(30, self._internal_motion)
        self.add_event(45, self._server_room_access)
    
    def _perimeter_motion(self, system):
        """Simulate motion detection at the perimeter"""
        logger.info("Event: Perimeter motion detected")
        
        motion_sensor = system.sensor_manager.get_sensor("motion1")
        if motion_sensor:
            motion_sensor.receive_message(Message(
                sender="simulation",
                receiver=motion_sensor.id,
                message_type=MessageType.EVENT,
                content={
                    "motion_detected": True,
                    "location": "Perimeter"
                }
            ))
    
    def _door_sensor_trigger(self, system):
        """Simulate a door sensor being triggered"""
        logger.info("Event: Door sensor triggered")
        
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="door_sensor",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "door_opened": True,
                    "door_id": "side_entrance",
                    "authorized": False,
                    "after_hours": True
                }
            ))
    
    def _camera_motion_detection(self, system):
        """Simulate motion detection on security camera"""
        logger.info("Event: Motion detected on security camera")
        
        camera_sensor = system.sensor_manager.get_sensor("camera1")
        if camera_sensor:
            camera_sensor.receive_message(Message(
                sender="simulation",
                receiver=camera_sensor.id,
                message_type=MessageType.EVENT,
                content={
                    "motion_detected": True,
                    "face_detected": True,
                    "recognized_face": None,
                    "location": "Side Entrance Hallway"
                }
            ))
    
    def _internal_motion(self, system):
        """Simulate motion detection inside the building"""
        logger.info("Event: Internal motion detected")
        
        motion_sensor = system.sensor_manager.get_sensor("motion1")
        if motion_sensor:
            motion_sensor.receive_message(Message(
                sender="simulation",
                receiver=motion_sensor.id,
                message_type=MessageType.EVENT,
                content={
                    "motion_detected": True,
                    "location": "Main Hallway"
                }
            ))
    
    def _server_room_access(self, system):
        """Simulate unauthorized access to server room"""
        logger.info("Event: Server room access detected")
        
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="door_sensor",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "door_opened": True,
                    "door_id": "server_room",
                    "authorized": False,
                    "after_hours": True,
                    "forced_entry": True
                }
            ))

class InternalThreatScenario(SimulationScenario):
    """Simulates an internal threat scenario with unusual employee behavior"""
    
    def __init__(self):
        super().__init__(
            "Internal Threat",
            "Simulates a scenario where an employee exhibits suspicious behavior"
        )
        
        # Add events to the scenario
        self.add_event(0, self._unusual_login_time)
        self.add_event(15, self._access_unusual_files)
        self.add_event(30, self._download_sensitive_data)
        self.add_event(45, self._unusual_email_communication)
        self.add_event(60, self._upload_to_external_site)
    
    def _unusual_login_time(self, system):
        """Simulate login at unusual hours"""
        logger.info("Event: Login at unusual hours")
        
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="auth_system",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "access_attempt": True,
                    "authorized": True,
                    "user": "john.doe",
                    "resource": "vpn",
                    "time": "03:15 AM",
                    "unusual_time": True
                }
            ))
    
    def _access_unusual_files(self, system):
        """Simulate access to unusual files"""
        logger.info("Event: Access to unusual files")
        
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="file_system",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "file_access": True,
                    "file_path": "/hr/salary_data.xlsx",
                    "user": "john.doe",
                    "access_type": "read",
                    "unusual_access": True,
                    "department_mismatch": True
                }
            ))
    
    def _download_sensitive_data(self, system):
        """Simulate download of sensitive data"""
        logger.info("Event: Download of sensitive data")
        
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="file_system",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "file_access": True,
                    "file_path": "/finance/quarterly_report_draft.pdf",
                    "user": "john.doe",
                    "access_type": "download",
                    "file_size": "25MB",
                    "unusual_access": True,
                    "department_mismatch": True
                }
            ))
    
    def _unusual_email_communication(self, system):
        """Simulate unusual email communication"""
        logger.info("Event: Unusual email communication detected")
        
        text_analyzer = system.sensor_manager.get_sensor("text1")
        if text_analyzer:
            text_analyzer.add_text(
                "I've gathered the files you requested. Will send them via the alternate channel we discussed. No one suspects anything.",
                "email_from_john.doe"
            )
    
    def _upload_to_external_site(self, system):
        """Simulate upload to external site"""
        logger.info("Event: Upload to external site detected")
        
        network_sensor = system.sensor_manager.get_sensor("network1")
        if network_sensor:
            network_sensor.receive_message(Message(
                sender="simulation",
                receiver=network_sensor.id,
                message_type=MessageType.EVENT,
                content={
                    "traffic_level": 0.6,
                    "suspicious_activity": True,
                    "upload_detected": True,
                    "destination": "unknown-file-sharing.com",
                    "file_size": "25MB",
                    "user": "john.doe",
                    "anomaly_score": 0.85,
                    "alert": True
                }
            ))

class CyberAttackScenario(SimulationScenario):
    """Simulates a sophisticated cyber attack scenario"""
    
    def __init__(self):
        super().__init__(
            "Sophisticated Cyber Attack",
            "Simulates a scenario with a multi-stage cyber attack against the organization"
        )
        
        # Add events to the scenario
        self.add_event(0, self._initial_reconnaissance)
        self.add_event(10, self._phishing_email)
        self.add_event(25, self._malware_download)
        self.add_event(40, self._command_and_control)
        self.add_event(55, self._lateral_movement)
        self.add_event(70, self._data_exfiltration)
    
    def _initial_reconnaissance(self, system):
        """Simulate initial reconnaissance"""
        logger.info("Event: Initial reconnaissance detected")
        
        network_sensor = system.sensor_manager.get_sensor("network1")
        if network_sensor:
            network_sensor.receive_message(Message(
                sender="simulation",
                receiver=network_sensor.id,
                message_type=MessageType.EVENT,
                content={
                    "traffic_level": 0.3,
                    "suspicious_activity": True,
                    "port_scanning": True,
                    "anomaly_score": 0.4,
                    "alert": False
                }
            ))
    
    def _phishing_email(self, system):
        """Simulate a phishing email"""
        logger.info("Event: Phishing email detected")
        
        text_analyzer = system.sensor_manager.get_sensor("text1")
        if text_analyzer:
            text_analyzer.add_text(
                "URGENT: Your account will be terminated. Click here to verify your credentials immediately: http://securityupdate.company-portal.net/login",
                "email_to_employees"
            )
    
    def _malware_download(self, system):
        """Simulate malware download"""
        logger.info("Event: Malware download detected")
        
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="endpoint_protection",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "malware_detected": True,
                    "file_name": "security_update.exe",
                    "user": "jane.smith",
                    "workstation": "WS-FINANCE-12",
                    "threat_level": "high"
                }
            ))
    
    def _command_and_control(self, system):
        """Simulate command and control traffic"""
        logger.info("Event: Command and control traffic detected")
        
        network_sensor = system.sensor_manager.get_sensor("network1")
        if network_sensor:
            network_sensor.receive_message(Message(
                sender="simulation",
                receiver=network_sensor.id,
                message_type=MessageType.EVENT,
                content={
                    "traffic_level": 0.5,
                    "suspicious_activity": True,
                    "unknown_ip": "185.123.45.67",
                    "unusual_protocol": True,
                    "beaconing_pattern": True,
                    "anomaly_score": 0.8,
                    "alert": True
                }
            ))
    
    def _lateral_movement(self, system):
        """Simulate lateral movement in the network"""
        logger.info("Event: Lateral movement detected")
        
        vigilance_agent = system.agent_system.get_agent("vigilance1")
        if vigilance_agent:
            vigilance_agent.receive_message(Message(
                sender="network_monitor",
                receiver=vigilance_agent.id,
                message_type=MessageType.EVENT,
                content={
                    "lateral_movement": True,
                    "source": "WS-FINANCE-12",
                    "destination": "FILE-SERVER-01",
                    "credential_use": "administrator",
                    "unusual_time": True
                }
            ))
    
    def _data_exfiltration(self, system):
        """Simulate data exfiltration"""
        logger.info("Event: Data exfiltration detected")
        
        network_sensor = system.sensor_manager.get_sensor("network1")
        if network_sensor:
            network_sensor.receive_message(Message(
                sender="simulation",
                receiver=network_sensor.id,
                message_type=MessageType.EVENT,
                content={
                    "traffic_level": 0.9,
                    "suspicious_activity": True,
                    "data_exfiltration": True,
                    "destination": "185.123.45.67",
                    "encrypted_channel": True,
                    "data_volume": "2.3GB",
                    "anomaly_score": 0.95,
                    "alert": True
                }
            ))

class ScenarioSimulator:
    """Manages and runs simulation scenarios"""
    
    def __init__(self):
        self.scenarios = {}  # name -> scenario
        self.active_scenarios = {}  # name -> scenario
    
    def register_scenario(self, scenario: SimulationScenario):
        """Register a scenario"""
        self.scenarios[scenario.name] = scenario
        logger.info(f"Registered scenario: {scenario.name}")
    
    def run_scenario(self, scenario_name: str, system) -> bool:
        """Run a scenario by name"""
        if scenario_name not in self.scenarios:
            logger.error(f"Unknown scenario: {scenario_name}")
            return False
        
        scenario = self.scenarios[scenario_name]
        
        if scenario.name in self.active_scenarios:
            logger.warning(f"Scenario '{scenario.name}' is already running")
            return False
        
        # Run the scenario
        scenario.run(system)
        self.active_scenarios[scenario.name] = scenario
        
        return True
    
    def stop_scenario(self, scenario_name: str) -> bool:
        """Stop a running scenario"""
        if scenario_name not in self.active_scenarios:
            logger.warning(f"Scenario '{scenario_name}' is not running")
            return False
        
        scenario = self.active_scenarios[scenario_name]
        scenario.stop()
        del self.active_scenarios[scenario_name]
        
        return True
    
    def stop_all_scenarios(self):
        """Stop all running scenarios"""
        for name in list(self.active_scenarios.keys()):
            self.stop_scenario(name)
    
    def get_available_scenarios(self) -> List[Dict[str, str]]:
        """Get a list of available scenarios"""
        return [
            {
                "name": scenario.name,
                "description": scenario.description,
                "active": scenario.name in self.active_scenarios
            }
            for scenario in self.scenarios.values()
        ]
    
    def get_active_scenarios(self) -> List[str]:
        """Get a list of active scenario names"""
        return list(self.active_scenarios.keys())

def create_default_simulator():
    """Create a simulator with default scenarios"""
    simulator = ScenarioSimulator()
    
    # Register standard scenarios
    simulator.register_scenario(UnauthorizedAccessScenario())
    simulator.register_scenario(PhysicalSecurityScenario())
    simulator.register_scenario(InternalThreatScenario())
    simulator.register_scenario(CyberAttackScenario())
    
    return simulator

# Command-line simulator interface
if __name__ == "__main__":
    import argparse
    import importlib.util
    import sys
    
    parser = argparse.ArgumentParser(description="Godfrey Scenario Simulator")
    parser.add_argument("--system", "-s", help="Path to system module (if not using main)")
    parser.add_argument("--list", "-l", action="store_true", help="List available scenarios")
    parser.add_argument("--run", "-r", help="Run a specific scenario")
    args = parser.parse_args()
    
    # Create simulator with default scenarios
    simulator = create_default_simulator()
    
    # List scenarios if requested
    if args.list:
        print("Available scenarios:")
        for scenario in simulator.get_available_scenarios():
            print(f"- {scenario['name']}")
            print(f"  {scenario['description']}")
            print()
        sys.exit(0)
    
    # Import system module if provided
    if args.system:
        try:
            spec = importlib.util.spec_from_file_location("system_module", args.system)
            system_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(system_module)
            system = system_module.create_system()
        except Exception as e:
            print(f"Error loading system module: {e}")
            sys.exit(1)
    else:
        # Try to import from the main module
        try:
            from godfrey_system.main import GodfredSystem
            system = GodfredSystem()
            system.setup()
            system.start()
        except Exception as e:
            print(f"Error creating system: {e}")
            sys.exit(1)
    
    # Run scenario if requested
    if args.run:
        try:
            if simulator.run_scenario(args.run, system):
                print(f"Running scenario: {args.run}")
                # Keep running until interrupted
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("Stopping scenario...")
                    simulator.stop_all_scenarios()
            else:
                print(f"Failed to run scenario: {args.run}")
        finally:
            # Stop the system
            system.stop()
    else:
        print("No scenario specified to run. Use --run or --list for available scenarios.")
        system.stop()