from typing import List, Tuple


class FewShotPrompting:
    """Few-shot and zero-shot prompting strategies."""

    def __init__(self, examples: List[Tuple[str, str]] = None):
        """
        Initialize few-shot prompting.
        Args:
            examples: List of (input, output) tuples for demonstrations.
        """
        self.examples = examples or []

    def create_zero_shot_prompt(self, query: str, system_msg: str = None) -> str:
        """
        Create a zero-shot prompt without examples.
        Args:
            query: User query
            system_msg: Optional system message
        Returns:
            Formatted prompt string
        """
        system = system_msg or "You are a helpful customer support assistant."
        return f"{system}\n\nUser: {query}\n\nAssistant:"

    def create_few_shot_prompt(
        self, query: str, num_examples: int = 2, system_msg: str = None
    ) -> str:
        """
        Create a few-shot prompt with examples.
        Args:
            query: User query
            num_examples: Number of examples to include
            system_msg: Optional system message
        Returns:
            Formatted prompt string
        """
        system = system_msg or "You are a helpful customer support assistant."
        prompt = f"{system}\n\n"

        # Add examples
        for in_text, out_text in self.examples[:num_examples]:
            prompt += f"Example:\nUser: {in_text}\nAssistant: {out_text}\n\n"

        prompt += f"User: {query}\n\nAssistant:"
        return prompt

    def add_example(self, input_text: str, output_text: str):
        """Add an example to the few-shot pool."""
        self.examples.append((input_text, output_text))

    def clear_examples(self):
        """Clear all examples."""
        self.examples = []
