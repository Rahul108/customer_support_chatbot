from typing import Dict, Optional


class RoleContextPrompting:
    """Role-specific and user-context based prompting."""

    ROLE_DEFINITIONS = {
        "customer_support_specialist": (
            "You are an expert customer support specialist with deep knowledge "
            "of our products and services. You are empathetic, patient, and focused "
            "on resolving issues quickly."
        ),
        "technical_support": (
            "You are a technical support engineer. You provide precise technical "
            "solutions with clear steps. You explain technical concepts clearly."
        ),
        "sales_representative": (
            "You are a friendly sales representative. You focus on understanding "
            "customer needs and recommending suitable products or services."
        ),
        "account_manager": (
            "You are an account manager. You prioritize customer satisfaction and "
            "long-term relationships. You consider overall customer value."
        ),
    }

    def __init__(self, default_role: str = "customer_support_specialist"):
        """
        Initialize role-context prompting.
        Args:
            default_role: Default role to use
        """
        self.current_role = default_role
        self.user_context = {}

    def set_role(self, role: str):
        """Set the current role."""
        if role not in self.ROLE_DEFINITIONS:
            raise ValueError(f"Unknown role: {role}")
        self.current_role = role

    def set_user_context(self, context: Dict[str, str]):
        """
        Set user context information.
        Args:
            context: Dictionary with user information (e.g., name, account_type, history)
        """
        self.user_context = context

    def get_role_prompt(self) -> str:
        """Get the role definition prompt."""
        return self.ROLE_DEFINITIONS.get(
            self.current_role, self.ROLE_DEFINITIONS["customer_support_specialist"]
        )

    def create_role_context_prompt(self, query: str, include_history: bool = False) -> str:
        """
        Create a prompt with role and user context.
        Args:
            query: User query
            include_history: Whether to include user history
        Returns:
            Formatted prompt string
        """
        prompt = f"{self.get_role_prompt()}\n\n"

        # Add user context
        if self.user_context:
            prompt += "User Context:\n"
            for key, value in self.user_context.items():
                if key != "history" or include_history:
                    prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
            prompt += "\n"

        prompt += f"Query: {query}\n\n"
        prompt += "Response:"

        return prompt

    def create_personalized_prompt(
        self, query: str, user_name: str = None, previous_issues: str = None
    ) -> str:
        """
        Create a personalized prompt based on user information.
        Args:
            query: User query
            user_name: User's name
            previous_issues: Summary of previous issues
        Returns:
            Formatted prompt string
        """
        prompt = f"{self.get_role_prompt()}\n\n"

        if user_name:
            prompt += f"Address the customer as {user_name}.\n"

        if previous_issues:
            prompt += f"The customer has previously reported: {previous_issues}\n"
            prompt += "Try to provide a solution that prevents similar issues.\n\n"

        prompt += f"Customer Query: {query}\n\n"
        prompt += "Response:"

        return prompt

    @staticmethod
    def create_multi_role_prompt(query: str, primary_role: str, secondary_roles: list = None) -> str:
        """
        Create a prompt combining multiple roles for comprehensive response.
        Args:
            query: User query
            primary_role: Primary role perspective
            secondary_roles: List of secondary role perspectives
        Returns:
            Formatted prompt string
        """
        prompt = f"Consider this query from multiple perspectives:\n\n"
        prompt += f"Primary Perspective ({primary_role}):\n"
        prompt += f"{RoleContextPrompting.ROLE_DEFINITIONS.get(primary_role, '')}\n\n"

        if secondary_roles:
            for role in secondary_roles:
                if role in RoleContextPrompting.ROLE_DEFINITIONS:
                    prompt += f"Secondary Perspective ({role}):\n"
                    prompt += f"{RoleContextPrompting.ROLE_DEFINITIONS[role]}\n\n"

        prompt += f"Query: {query}\n\n"
        prompt += "Comprehensive Response:"

        return prompt
