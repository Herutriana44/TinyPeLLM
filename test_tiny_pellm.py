"""
Test suite for TinyPeLLM implementation
"""

import unittest
import torch
import tempfile
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline

# Import our custom classes
from TinyPeLLMModel import TinyPeLLMConfig, TinyPeLLMForCausalLM
from TinyPeLLMTokenizer import TinyPeLLMTokenizer
from TinyPeLLMPipeline import register_tiny_pellm, TinyPeLLMTrainer, TinyPeLLMPipeline


class TestTinyPeLLMConfig(unittest.TestCase):
    """Test TinyPeLLMConfig class"""
    
    def test_config_creation(self):
        """Test config creation with default parameters"""
        config = TinyPeLLMConfig()
        self.assertEqual(config.vocab_size, 3000)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.model_type, "tiny_pellm")
    
    def test_config_custom_parameters(self):
        """Test config creation with custom parameters"""
        config = TinyPeLLMConfig(
            vocab_size=5000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8
        )
        self.assertEqual(config.vocab_size, 5000)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.num_attention_heads, 8)
    
    def test_config_save_load(self):
        """Test config save and load functionality"""
        config = TinyPeLLMConfig(vocab_size=1000, hidden_size=64)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_pretrained(temp_dir)
            loaded_config = TinyPeLLMConfig.from_pretrained(temp_dir)
            
            self.assertEqual(config.vocab_size, loaded_config.vocab_size)
            self.assertEqual(config.hidden_size, loaded_config.hidden_size)


class TestTinyPeLLMTokenizer(unittest.TestCase):
    """Test TinyPeLLMTokenizer class"""
    
    def test_tokenizer_creation(self):
        """Test tokenizer creation"""
        tokenizer = TinyPeLLMTokenizer()
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(tokenizer._tokenizer)
    
    def test_tokenizer_encode_decode(self):
        """Test basic encode and decode functionality"""
        # Use the pre-trained model file
        tokenizer = TinyPeLLMTokenizer()
        
        text = "Hello world"
        
        # Test encode
        encoded = tokenizer.encode(text)
        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 0)
        
        # Test decode
        decoded = tokenizer.decode(encoded)
        self.assertIsInstance(decoded, str)
        # Check if the decoded text contains meaningful content (not just special tokens)
        self.assertGreater(len(decoded.strip()), 0)
    
    def test_tokenizer_save_load(self):
        """Test tokenizer save and load functionality"""
        tokenizer = TinyPeLLMTokenizer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir)
            
            # Check if files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "tokenizer.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "vocab.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "tokenizer_config.json")))
            
            # Load tokenizer
            loaded_tokenizer = TinyPeLLMTokenizer.from_pretrained(temp_dir)
            self.assertIsNotNone(loaded_tokenizer)


class TestTinyPeLLMModel(unittest.TestCase):
    """Test TinyPeLLMModel class"""
    
    def test_model_creation(self):
        """Test model creation"""
        config = TinyPeLLMConfig(vocab_size=1000, hidden_size=64)
        model = TinyPeLLMForCausalLM(config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.vocab_size, 1000)
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        config = TinyPeLLMConfig(vocab_size=1000, hidden_size=64)
        model = TinyPeLLMForCausalLM(config)
        
        # Create dummy input
        batch_size, seq_length = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape, (batch_size, seq_length, 1000))
    
    def test_model_save_load(self):
        """Test model save and load functionality"""
        config = TinyPeLLMConfig(vocab_size=1000, hidden_size=64)
        model = TinyPeLLMForCausalLM(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, safe_serialization=False)
            
            # Check if files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "pytorch_model.bin")))
            
            # Load model
            loaded_model = TinyPeLLMForCausalLM.from_pretrained(temp_dir)
            self.assertIsNotNone(loaded_model)


class TestAutoRegistration(unittest.TestCase):
    """Test Auto class registration"""
    
    def test_registration(self):
        """Test that registration works correctly"""
        register_tiny_pellm()
        
        # Test that config can be loaded
        config = TinyPeLLMConfig(vocab_size=1000, hidden_size=64)
        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_pretrained(temp_dir)
            
            # Should be able to load with AutoConfig
            loaded_config = AutoConfig.from_pretrained(temp_dir)
            self.assertEqual(loaded_config.model_type, "tiny_pellm")
    
    def test_auto_tokenizer(self):
        """Test AutoTokenizer integration"""
        register_tiny_pellm()
        
        tokenizer = TinyPeLLMTokenizer()
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir)
            
            # Should be able to load with AutoTokenizer
            loaded_tokenizer = AutoTokenizer.from_pretrained(temp_dir)
            self.assertIsNotNone(loaded_tokenizer)
    
    def test_auto_model(self):
        """Test AutoModelForCausalLM integration"""
        register_tiny_pellm()
        
        config = TinyPeLLMConfig(vocab_size=1000, hidden_size=64)
        model = TinyPeLLMForCausalLM(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, safe_serialization=False)
            
            # Should be able to load with AutoModelForCausalLM
            loaded_model = AutoModelForCausalLM.from_pretrained(temp_dir)
            self.assertIsNotNone(loaded_model)


class TestPipeline(unittest.TestCase):
    """Test pipeline functionality"""
    
    def test_pipeline_creation(self):
        """Test pipeline creation"""
        register_tiny_pellm()
        
        # Create model and tokenizer
        config = TinyPeLLMConfig(vocab_size=1000, hidden_size=64)
        tokenizer = TinyPeLLMTokenizer()
        model = TinyPeLLMForCausalLM(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, safe_serialization=False)
            tokenizer.save_pretrained(temp_dir)
            
            # Create pipeline
            pipe = pipeline("text-generation", model=temp_dir, tokenizer=temp_dir)
            self.assertIsNotNone(pipe)
    
    def test_custom_pipeline(self):
        """Test custom pipeline class"""
        register_tiny_pellm()
        
        # Create model and tokenizer
        config = TinyPeLLMConfig(vocab_size=1000, hidden_size=64)
        tokenizer = TinyPeLLMTokenizer()
        model = TinyPeLLMForCausalLM(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, safe_serialization=False)
            tokenizer.save_pretrained(temp_dir)
            
            # Create custom pipeline
            custom_pipe = TinyPeLLMPipeline(temp_dir)
            self.assertIsNotNone(custom_pipe)


class TestTrainer(unittest.TestCase):
    """Test trainer functionality"""
    
    def test_trainer_creation(self):
        """Test trainer creation"""
        register_tiny_pellm()
        trainer = TinyPeLLMTrainer()
        self.assertIsNotNone(trainer)
    
    def test_model_preparation(self):
        """Test model and tokenizer preparation"""
        register_tiny_pellm()
        trainer = TinyPeLLMTrainer()
        
        model, tokenizer = trainer.prepare_model_and_tokenizer(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=1
        )
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        self.assertEqual(model.config.vocab_size, 1000)
        self.assertEqual(model.config.hidden_size, 64)


class TestIntegration(unittest.TestCase):
    """Test full integration"""
    
    def test_full_workflow(self):
        """Test complete workflow from creation to inference"""
        register_tiny_pellm()
        
        # Create model and tokenizer
        config = TinyPeLLMConfig(vocab_size=1000, hidden_size=64)
        tokenizer = TinyPeLLMTokenizer()
        model = TinyPeLLMForCausalLM(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model and tokenizer
            model.save_pretrained(temp_dir, safe_serialization=False)
            tokenizer.save_pretrained(temp_dir)
            
            # Load with Auto classes
            loaded_tokenizer = AutoTokenizer.from_pretrained(temp_dir)
            loaded_model = AutoModelForCausalLM.from_pretrained(temp_dir)
            
            # Test forward pass
            text = "Hello"
            inputs = loaded_tokenizer(text, return_tensors="pt")
            outputs = loaded_model(**inputs)
            
            self.assertIsNotNone(outputs.logits)
            self.assertEqual(outputs.logits.shape[0], 1)  # batch size
            self.assertEqual(outputs.logits.shape[-1], 1000)  # vocab size


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTinyPeLLMConfig,
        TestTinyPeLLMTokenizer,
        TestTinyPeLLMModel,
        TestAutoRegistration,
        TestPipeline,
        TestTrainer,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running TinyPeLLM tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    exit(0 if success else 1) 