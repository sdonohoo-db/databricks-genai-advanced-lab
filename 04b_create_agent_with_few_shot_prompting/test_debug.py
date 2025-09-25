#!/usr/bin/env python3

import sys
sys.path.append('/Users/scott.donohoo/projects/databricks-genai-advanced-lab-main/04b_create_agent_with_few_shot_prompting')

from agent import AGENT
import json

# Test the agent with a simple request
test_request = {
    "csv_files": [
        "/Users/scott.donohoo/projects/databricks-genai-advanced-lab-main/04b_create_agent_with_few_shot_prompting/stories_authentication.csv"
    ],
    "input_template": {
        "issue_id": "AUTH-101",
        "description": "As a user, I want to register a new account so that I can access the platform",
        "acceptance_criteria": ["User can enter email and password", "System validates email format"]
    },
    "output_template": {
        "test_case_id": "AUTH-101",
        "test_title": "Test user registration with valid credentials",
        "test_description": "Verify that a user can successfully register a new account with valid email and password",
        "test_steps": [
            "Navigate to registration page",
            "Enter valid email address",
            "Enter valid password",
            "Click register button"
        ],
        "expected_result": "User account is created successfully and confirmation email is sent"
    }
}

print("=== Testing Agent with Debug Output ===")

# Create a mock request object
class MockRequest:
    def __init__(self, input_data):
        self.input = [MockMessage(json.dumps(input_data))]
        self.custom_inputs = {}

class MockMessage:
    def __init__(self, content):
        self.content = content
        self.role = "user"

    def model_dump(self):
        return {"role": self.role, "content": self.content}

mock_request = MockRequest(test_request)

try:
    result = AGENT.predict(mock_request)
    print("\n=== Agent Result ===")
    print("Result type:", type(result))
    print("Result attributes:", dir(result))

    if hasattr(result, 'output'):
        print("Output length:", len(result.output))
        if result.output:
            print("First output:", result.output[0])

    print("\n=== Now testing DataFrame conversion ===")

    # Import the conversion function from the notebook
    exec(open('/Users/scott.donohoo/projects/databricks-genai-advanced-lab-main/04b_create_agent_with_few_shot_prompting/04b_create_agent_with_few_shot_prompting.py').read())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()