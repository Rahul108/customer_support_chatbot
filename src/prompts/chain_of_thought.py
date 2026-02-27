from typing import List


class ChainOfThoughtPrompting:
    """Chain-of-Thought prompting for step-by-step reasoning."""

    def __init__(self, temperature: float = 0.7):
        """
        Initialize chain-of-thought prompting.
        Args:
            temperature: Sampling temperature for reasoning steps.
        """
        self.temperature = temperature

    def create_cot_prompt(
        self, query: str, reasoning_steps: List[str] = None
    ) -> str:
        """
        Create a chain-of-thought prompt.
        Args:
            query: User query
            reasoning_steps: Optional pre-defined reasoning steps
        Returns:
            Formatted prompt string
        """
        prompt = f"Let's think step-by-step.\n\n"
        prompt += f"Question: {query}\n\n"
        prompt += "Step-by-step reasoning:\n"

        if reasoning_steps:
            for i, step in enumerate(reasoning_steps, 1):
                prompt += f"{i}. {step}\n"
            prompt += "\nFinal Answer:"
        else:
            prompt += (
                "1. [First, understand what the customer is asking for]\n"
            )
            prompt += "2. [Identify the key issue]\n"
            prompt += "3. [Consider relevant solutions]\n"
            prompt += "4. [Provide the best solution]\n\nFinal Answer:"

        return prompt

    def create_structured_cot_prompt(
        self, query: str, context: str = None
    ) -> str:
        """
        Create a structured chain-of-thought prompt.
        Args:
            query: User query
            context: Optional context information
        Returns:
            Formatted prompt string
        """
        prompt = f"Analyze the customer issue following this structure:\n\n"

        if context:
            prompt += f"Context: {context}\n\n"

        prompt += f"Customer Query: {query}\n\n"
        prompt += "Analysis:\n"
        prompt += "1. Problem Understanding:\n"
        prompt += "2. Root Cause:\n"
        prompt += "3. Possible Solutions:\n"
        prompt += "4. Recommended Solution:\n"
        prompt += "5. Implementation Steps:\n"
        prompt += "\nResponse:"

        return prompt
