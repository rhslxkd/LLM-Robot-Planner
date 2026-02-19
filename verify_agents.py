import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

# Mock the config module BEFORE importing agents
sys.modules['vlm_courtroom.config'] = MagicMock()
sys.modules['vlm_courtroom.config'].get_model = MagicMock(return_value=MagicMock())

from vlm_courtroom.agents.specific_agents import JudgeAgent

def test_judge_agent():
    print("Initializing JudgeAgent...")
    try:
        judge = JudgeAgent()
        print("JudgeAgent initialized.")
        
        # Test process method to ensure prompt generation doesn't crash
        print("Testing process method prompt generation...")
        # We don't need real contact with LLM, just need to see if f-string evaluates
        # But generate_response calls the LLM. 
        # We can mock generate_response or just catch the error if it's not the ValueError.
        
        # Monkey patch generate_response to avoid actual API call
        judge.generate_response = lambda prompt, **kwargs: "Mock Response"
        
        judge.process({
            'original_proposal': 'orig',
            'prosecution_argument': 'pros',
            'defense_argument': 'def'
        })
        print("✅ JudgeAgent.process() executed successfully (Prompt f-string is valid).")
        
    except ValueError as e:
        print(f"❌ ValueError caught: {e}")
        print("This likely means the f-string in JudgeAgent.process has invalid format.")
    except Exception as e:
        print(f"⚠️ Other error caught: {e}")

if __name__ == "__main__":
    test_judge_agent()
