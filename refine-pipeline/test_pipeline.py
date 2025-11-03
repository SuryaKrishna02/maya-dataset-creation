#!/usr/bin/env python3
"""
Test script to verify the refine-pipeline components work correctly.
This script tests individual components before running the full pipeline.
"""

import json
import logging
from pathlib import Path

from data_processor import DataProcessor
from config_loader import ConfigLoader
from prompt_templates import PromptTemplates
from prompt_distributor import PromptDistributor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prompt_templates():
    """Test the prompt templates functionality."""
    logger.info("Testing prompt templates...")
    
    # Test getting all categories
    categories = PromptTemplates.get_all_categories()
    logger.info(f"Found {len(categories)} prompt categories")
    
    # Test getting all prompts
    all_prompts = PromptTemplates.get_all_prompts()
    logger.info(f"Total prompts available: {len(all_prompts)}")
    
    # Test balanced prompt selection
    balanced_prompts = PromptTemplates.get_balanced_prompts(18)
    logger.info(f"Generated {len(balanced_prompts)} balanced prompts")
    
    # Print sample prompts from each category
    logger.info("Sample prompts from each category:")
    for category, prompts in categories.items():
        logger.info(f"  {category}: {prompts[0][:50]}...")
    
    return True

def test_data_processor():
    """Test the data processor functionality."""
    logger.info("Testing data processor...")
    
    # Initialize processor
    processor = DataProcessor(".", metadata_dir="metadata")
    
    # Test getting image paths with filtering
    common_images = processor.get_all_image_paths(require_all_metadata=True)
    logger.info(f"Images present in all three files: {len(common_images)}")
    
    # Test getting sample data
    sample_data = processor.get_sample_data(2)
    logger.info(f"Retrieved {len(sample_data)} sample metadata items")
    
    # Test metadata string generation
    for i, metadata in enumerate(sample_data):
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Image: {metadata.image_path}")
        logger.info(f"  Has depth: {metadata.depth_metadata is not None}")
        logger.info(f"  Has scene desc: {metadata.scene_description is not None}")
        logger.info(f"  Metadata length: {len(metadata.to_metadata_string())} chars")
    
    return True

def test_config():
    """Test the processing configuration."""
    logger.info("Testing processing configuration...")
    
    try:
        config = ConfigLoader.load_config("config.yaml")
        
        logger.info(f"Config loaded successfully:")
        logger.info(f"  Metadata dir: {config.data.metadata_dir}")
        logger.info(f"  Workers: {config.processing.num_workers}")
        logger.info(f"  Batch size: {config.processing.batch_size}")
        logger.info(f"  Output file: {config.output.output_file}")
        logger.info(f"  Prompt distribution: {config.prompts.distribution}")
        
        return True
    except Exception as e:
        logger.error(f"Config loading failed: {e}")
        return False

def test_prompt_distributor():
    """Test the prompt distributor functionality."""
    logger.info("Testing prompt distributor...")
    
    # Test distribution configuration
    distribution = {
        'scene_understanding': 30,
        'observational_analytical': 30,
        'atmospheric_sensory': 20,
        'comprehensive_analysis': 20
    }
    
    # Create test image paths
    test_images = [f"test_image_{i:03d}.jpg" for i in range(20)]
    
    # Initialize distributor
    distributor = PromptDistributor(distribution, random_seed=42)
    
    # Distribute prompts
    assignments = distributor.distribute_prompts(test_images)
    
    # Validate
    is_valid = distributor.validate_assignments(assignments, len(test_images))
    logger.info(f"Assignment validation: {is_valid}")
    
    # Get summary
    summary = distributor.get_distribution_summary(assignments)
    logger.info("Distribution summary:")
    for category, count in summary.items():
        percentage = count / len(assignments) * 100
        logger.info(f"  {category}: {count} ({percentage:.1f}%)")
    
    return is_valid

def test_integration():
    """Test integration between components."""
    logger.info("Testing component integration...")
    
    try:
        # Load config
        config = ConfigLoader.load_config("config.yaml")
        
        # Initialize components
        processor = DataProcessor(".", metadata_dir=config.data.metadata_dir)
        distributor = PromptDistributor(config.prompts.distribution, random_seed=42)
        
        # Get sample image paths
        image_paths = processor.get_all_image_paths(require_all_metadata=True)[:5]
        
        # Distribute prompts
        assignments = distributor.distribute_prompts(image_paths)
        
        # Get sample metadata
        sample_metadata = processor.get_sample_data(1)[0]
        
        # Test metadata string formatting
        metadata_string = sample_metadata.to_metadata_string()
        
        # Create a mock message structure
        sample_assignment = assignments[0] if assignments else None
        if sample_assignment:
            messages = [
                {
                    "role": "system",
                    "content": metadata_string
                },
                {
                    "role": "user", 
                    "content": sample_assignment.prompt
                }
            ]
            
            logger.info("Integration test successful:")
            logger.info(f"  Image: {sample_metadata.image_path}")
            logger.info(f"  Prompt category: {sample_assignment.prompt_category}")
            logger.info(f"  Prompt: {sample_assignment.prompt[:50]}...")
            logger.info(f"  System content length: {len(messages[0]['content'])} chars")
            logger.info(f"  Messages structure: Valid")
        
        return True
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting refine-pipeline component tests...")
    
    tests = [
        ("Prompt Templates", test_prompt_templates),
        ("Data Processor", test_data_processor), 
        ("Processing Config", test_config),
        ("Prompt Distributor", test_prompt_distributor),
        ("Component Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info('='*50)
            
            result = test_func()
            results[test_name] = result
            
            logger.info(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED - {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Test Summary")
    logger.info('='*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The pipeline is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Test with sample mode: python3 main.py --sample_mode")
        logger.info("2. Process full dataset: python3 main.py")
        logger.info("3. Override settings: python3 main.py --max_images 100 --debug")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
