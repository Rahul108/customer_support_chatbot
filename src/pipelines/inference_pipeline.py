import torch
from typing import Optional, Dict
from config.settings import Config
from src.models.base_model import BaseModel
from src.prompts.few_shot import FewShotPrompting
from src.prompts.chain_of_thought import ChainOfThoughtPrompting
from src.prompts.role_context import RoleContextPrompting


class InferencePipeline:
    """Inference pipeline for generating customer support responses."""

    def __init__(self, model: BaseModel, config: Config):
        self.model = model
        self.config = config
        self.few_shot = FewShotPrompting()
        self.cot = ChainOfThoughtPrompting()
        self.role_context = RoleContextPrompting()

    def generate_response(
        self,
        query: str,
        strategy: str = "few_shot",
        user_context: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response for a customer query.
        Args:
            query: Customer query
            strategy: Prompting strategy ('few_shot', 'cot', 'role_context', 'combined')
            user_context: Optional user context information
            **kwargs: Additional arguments
        Returns:
            Generated response
        """
        prompt = self._build_prompt(query, strategy, user_context, **kwargs)
        response = self.model.generate(prompt, max_length=self.config.model.max_length)
        return self._extract_response(response)

    def _build_prompt(
        self,
        query: str,
        strategy: str,
        user_context: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        """Build prompt based on strategy."""
        if strategy == "few_shot":
            return self.few_shot.create_few_shot_prompt(
                query,
                num_examples=self.config.prompt.few_shot_examples,
                system_msg=kwargs.get("system_msg"),
            )
        elif strategy == "cot":
            return self.cot.create_structured_cot_prompt(
                query, context=kwargs.get("context")
            )
        elif strategy == "role_context":
            if user_context:
                self.role_context.set_user_context(user_context)
            return self.role_context.create_role_context_prompt(
                query, include_history=kwargs.get("include_history", False)
            )
        elif strategy == "combined":
            role_prompt = self.role_context.create_role_context_prompt(query)
            cot_prompt = self.cot.create_structured_cot_prompt(query)
            few_shot_prompt = self.few_shot.create_few_shot_prompt(
                query, num_examples=1
            )
            return f"{role_prompt}\n\n{cot_prompt}\n\nContext:\n{few_shot_prompt}"
        else:
            return self.few_shot.create_zero_shot_prompt(query)

    def _extract_response(self, generated_text: str) -> str:
        """Extract response from generated text."""
        if "Assistant:" in generated_text:
            return generated_text.split("Assistant:")[-1].strip()
        elif "Response:" in generated_text:
            return generated_text.split("Response:")[-1].strip()
        else:
            return generated_text.strip()

    def add_few_shot_example(self, input_text: str, output_text: str):
        """Add an example for few-shot learning."""
        self.few_shot.add_example(input_text, output_text)

    def set_role(self, role: str):
        """Set the current role for role-based prompting."""
        self.role_context.set_role(role)

    def batch_generate(
        self,
        queries: list,
        strategy: str = "few_shot",
        user_contexts: list = None,
    ) -> list:
        """
        Generate responses for multiple queries.
        Args:
            queries: List of customer queries
            strategy: Prompting strategy
            user_contexts: Optional list of user contexts
        Returns:
            List of generated responses
        """
        responses = []
        for i, query in enumerate(queries):
            user_context = user_contexts[i] if user_contexts else None
            response = self.generate_response(query, strategy, user_context)
            responses.append(response)
        return responses
