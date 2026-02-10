
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from vlm_courtroom.config import get_model

class Message:
    def __init__(self, sender: str, content: str, role: str = "user"):
        self.sender = sender
        self.content = content
        self.role = role # 'user', 'model', or specific agent role

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "content": self.content,
            "role": self.role
        }

class VLMAgent(ABC):
    def __init__(self, name: str, role: str, model_role: str = "DEFAULT"):
        self.name = name
        self.role = role
        self.model_role = model_role
        self.model = get_model(model_role)
        self.memory: List[Message] = []

    def add_to_memory(self, message: Message):
        self.memory.append(message)

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Message:
        """
        Process the current context and return a response Message.
        Context can include the image, previous messages, etc.
        """
        pass

    def generate_response(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Helper to generate content from Gemini."""
        try:
            # TODO: Add image handling if image_path is provided
            # For now, text-only or text-heavy prompts
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"
