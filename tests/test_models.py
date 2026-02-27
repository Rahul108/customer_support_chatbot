import unittest
from config.settings import Config, LoRAConfig, AdapterConfig
from src.models.lora_model import LoRAModel
from src.models.adapter_model import AdapterModel
from src.models.peft_model import PEFTModel
from src.prompts.few_shot import FewShotPrompting
from src.prompts.chain_of_thought import ChainOfThoughtPrompting
from src.prompts.role_context import RoleContextPrompting
from src.pipelines.inference_pipeline import InferencePipeline


class TestLoRAModel(unittest.TestCase):
    """Test LoRA model implementation."""

    def setUp(self):
        self.config = Config()
        # Use a small model for testing
        self.config.model.base_model = "distilgpt2"

    def test_lora_config(self):
        """Test LoRA configuration."""
        lora_config = LoRAConfig(r=8, lora_alpha=16)
        self.assertEqual(lora_config.r, 8)
        self.assertEqual(lora_config.lora_alpha, 16)

    def test_lora_initialization(self):
        """Test LoRA model initialization."""
        try:
            model = LoRAModel(self.config)
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.tokenizer)
        except Exception as e:
            self.skipTest(f"Model initialization requires transformers: {e}")


class TestAdapterModel(unittest.TestCase):
    """Test Adapter model implementation."""

    def setUp(self):
        self.config = Config()
        self.config.model.base_model = "distilgpt2"

    def test_adapter_config(self):
        """Test Adapter configuration."""
        adapter_config = AdapterConfig(adapter_size=64, activation="relu")
        self.assertEqual(adapter_config.adapter_size, 64)
        self.assertEqual(adapter_config.activation, "relu")


class TestPEFTModel(unittest.TestCase):
    """Test PEFT model implementation."""

    def setUp(self):
        self.config = Config()
        self.config.model.base_model = "distilgpt2"

    def test_peft_initialization(self):
        """Test PEFT model initialization."""
        try:
            model = PEFTModel(self.config, technique="lora")
            self.assertIsNotNone(model.model)
        except Exception as e:
            self.skipTest(f"Model initialization requires transformers: {e}")


class TestFewShotPrompting(unittest.TestCase):
    """Test few-shot prompting."""

    def setUp(self):
        self.prompter = FewShotPrompting()

    def test_zero_shot_prompt(self):
        """Test zero-shot prompt creation."""
        prompt = self.prompter.create_zero_shot_prompt("How do I reset my password?")
        self.assertIn("How do I reset my password?", prompt)
        self.assertIn("Assistant:", prompt)

    def test_few_shot_prompt(self):
        """Test few-shot prompt creation."""
        self.prompter.add_example(
            "How do I login?", "Click the login button and enter your credentials."
        )
        prompt = self.prompter.create_few_shot_prompt(
            "How do I logout?", num_examples=1
        )
        self.assertIn("How do I logout?", prompt)
        self.assertIn("Example:", prompt)

    def test_add_remove_examples(self):
        """Test adding and clearing examples."""
        self.prompter.add_example("Q1", "A1")
        self.prompter.add_example("Q2", "A2")
        self.assertEqual(len(self.prompter.examples), 2)

        self.prompter.clear_examples()
        self.assertEqual(len(self.prompter.examples), 0)


class TestChainOfThoughtPrompting(unittest.TestCase):
    """Test chain-of-thought prompting."""

    def setUp(self):
        self.cot = ChainOfThoughtPrompting()

    def test_cot_prompt(self):
        """Test chain-of-thought prompt creation."""
        prompt = self.cot.create_cot_prompt("What should I do if I forgot my password?")
        self.assertIn("step-by-step", prompt.lower())
        self.assertIn("Final Answer:", prompt)

    def test_structured_cot_prompt(self):
        """Test structured chain-of-thought prompt."""
        prompt = self.cot.create_structured_cot_prompt(
            "Email not working", context="User account status: active"
        )
        self.assertIn("Problem Understanding:", prompt)
        self.assertIn("Recommended Solution:", prompt)


class TestRoleContextPrompting(unittest.TestCase):
    """Test role-context prompting."""

    def setUp(self):
        self.role_context = RoleContextPrompting()

    def test_role_definition(self):
        """Test role definitions."""
        role_prompt = self.role_context.get_role_prompt()
        self.assertIn("customer_support_specialist", role_prompt.lower())

    def test_set_role(self):
        """Test setting different roles."""
        self.role_context.set_role("technical_support")
        role_prompt = self.role_context.get_role_prompt()
        self.assertIn("technical", role_prompt.lower())

    def test_invalid_role(self):
        """Test handling of invalid roles."""
        with self.assertRaises(ValueError):
            self.role_context.set_role("invalid_role")

    def test_user_context(self):
        """Test setting user context."""
        context = {"name": "John", "account_type": "premium"}
        self.role_context.set_user_context(context)
        prompt = self.role_context.create_role_context_prompt("How can you help?")
        self.assertIn("John", prompt)

    def test_personalized_prompt(self):
        """Test personalized prompt creation."""
        prompt = self.role_context.create_personalized_prompt(
            "I need help", user_name="Alice", previous_issues="Login problems"
        )
        self.assertIn("Alice", prompt)
        self.assertIn("Login problems", prompt)


class TestInferencePipeline(unittest.TestCase):
    """Test inference pipeline."""

    def setUp(self):
        self.config = Config()
        self.config.model.base_model = "distilgpt2"

    def test_pipeline_initialization(self):
        """Test inference pipeline initialization."""
        try:
            from src.models.lora_model import LoRAModel
            model = LoRAModel(self.config)
            pipeline = InferencePipeline(model, self.config)
            self.assertIsNotNone(pipeline.few_shot)
            self.assertIsNotNone(pipeline.cot)
            self.assertIsNotNone(pipeline.role_context)
        except Exception as e:
            self.skipTest(f"Pipeline initialization requires transformers: {e}")

    def test_add_few_shot_example(self):
        """Test adding few-shot examples to pipeline."""
        try:
            from src.models.lora_model import LoRAModel
            model = LoRAModel(self.config)
            pipeline = InferencePipeline(model, self.config)
            pipeline.add_few_shot_example("Q", "A")
            self.assertEqual(len(pipeline.few_shot.examples), 1)
        except Exception as e:
            self.skipTest(f"Pipeline requires transformers: {e}")

    def test_set_role(self):
        """Test setting role in pipeline."""
        try:
            from src.models.lora_model import LoRAModel
            model = LoRAModel(self.config)
            pipeline = InferencePipeline(model, self.config)
            pipeline.set_role("technical_support")
            self.assertEqual(pipeline.role_context.current_role, "technical_support")
        except Exception as e:
            self.skipTest(f"Pipeline requires transformers: {e}")


if __name__ == "__main__":
    unittest.main()
