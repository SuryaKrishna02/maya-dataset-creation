"""
Prompt distributor for percentage-based prompt assignment across the dataset.
Ensures each image gets exactly one prompt based on configured distribution percentages.
"""

import random
import logging
from typing import List, Dict
from dataclasses import dataclass

from prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

@dataclass
class PromptAssignment:
    """Represents a prompt assignment for an image."""
    image_path: str
    prompt: str
    prompt_category: str

class PromptDistributor:
    """Handles percentage-based distribution of prompts across the dataset."""
    
    def __init__(self, distribution: Dict[str, int], random_seed: int = None, shuffle_dataset: bool = True):
        """
        Initialize the prompt distributor.
        
        Args:
            distribution: Dictionary mapping prompt categories to percentages
            random_seed: Random seed for reproducible results
            shuffle_dataset: Whether to shuffle the dataset before assignment
        """
        self.distribution = distribution
        self.shuffle_dataset = shuffle_dataset
        
        # Validate distribution
        self._validate_distribution()
        
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            logger.info(f"Set random seed to {random_seed} for reproducible prompt assignment")
        
        # Get all available prompts by category
        self.prompts_by_category = PromptTemplates.get_all_categories()
        
        logger.info(f"Initialized prompt distributor with distribution: {distribution}")
    
    def _validate_distribution(self) -> None:
        """Validate the distribution configuration."""
        total_percentage = sum(self.distribution.values())
        if total_percentage != 100:
            raise ValueError(f"Distribution percentages must sum to 100, got {total_percentage}")
        
        # Check if all categories exist in prompt templates
        available_categories = set(PromptTemplates.get_all_categories().keys())
        configured_categories = set(self.distribution.keys())
        
        missing_categories = configured_categories - available_categories
        if missing_categories:
            raise ValueError(f"Unknown prompt categories: {missing_categories}")
        
        unused_categories = available_categories - configured_categories
        if unused_categories:
            logger.warning(f"Prompt categories not used in distribution: {unused_categories}")
    
    def _calculate_assignments(self, total_images: int) -> Dict[str, int]:
        """Calculate the number of images to assign to each category."""
        assignments = {}
        remaining_images = total_images
        
        # Sort categories by percentage (largest first) for better distribution
        sorted_categories = sorted(self.distribution.items(), key=lambda x: x[1], reverse=True)
        
        for i, (category, percentage) in enumerate(sorted_categories):
            if i == len(sorted_categories) - 1:
                # Last category gets all remaining images to ensure exact total
                count = remaining_images
            else:
                count = round(total_images * percentage / 100)
                remaining_images -= count
            
            assignments[category] = count
            logger.info(f"Category '{category}': {count} images ({count/total_images*100:.1f}%)")
        
        # Verify total
        total_assigned = sum(assignments.values())
        if total_assigned != total_images:
            logger.warning(f"Assignment mismatch: {total_assigned} assigned vs {total_images} total")
        
        return assignments
    
    def _select_prompts_for_category(self, category: str, count: int) -> List[str]:
        """Select prompts for a category, cycling through available prompts if needed."""
        category_prompts = self.prompts_by_category[category]
        selected_prompts = []
        
        for i in range(count):
            # Cycle through prompts if we need more than available
            prompt = category_prompts[i % len(category_prompts)]
            selected_prompts.append(prompt)
        
        return selected_prompts
    
    def distribute_prompts(self, image_paths: List[str]) -> List[PromptAssignment]:
        """
        Distribute prompts across images based on configured percentages.
        
        Args:
            image_paths: List of image paths to assign prompts to
            
        Returns:
            List of PromptAssignment objects
        """
        total_images = len(image_paths)
        logger.info(f"Distributing prompts across {total_images} images")
        
        # Make a copy of image paths to avoid modifying the original
        working_image_paths = image_paths.copy()
        
        # Shuffle if requested
        if self.shuffle_dataset:
            random.shuffle(working_image_paths)
            logger.info("Shuffled dataset for random prompt assignment")
        
        # Calculate assignments per category
        assignments = self._calculate_assignments(total_images)
        
        # Create prompt assignments
        prompt_assignments = []
        current_index = 0
        
        for category, count in assignments.items():
            if count == 0:
                continue
                
            # Get images for this category
            category_images = working_image_paths[current_index:current_index + count]
            
            # Select prompts for this category
            category_prompts = self._select_prompts_for_category(category, count)
            
            # Create assignments
            for image_path, prompt in zip(category_images, category_prompts):
                assignment = PromptAssignment(
                    image_path=image_path,
                    prompt=prompt,
                    prompt_category=category
                )
                prompt_assignments.append(assignment)
            
            current_index += count
            logger.info(f"Assigned {count} prompts for category '{category}'")
        
        logger.info(f"Created {len(prompt_assignments)} prompt assignments")
        return prompt_assignments
    
    def get_distribution_summary(self, assignments: List[PromptAssignment]) -> Dict[str, int]:
        """Get a summary of the actual distribution."""
        summary = {}
        for assignment in assignments:
            category = assignment.prompt_category
            summary[category] = summary.get(category, 0) + 1
        
        return summary
    
    def validate_assignments(self, assignments: List[PromptAssignment], expected_total: int) -> bool:
        """Validate that assignments match expectations."""
        actual_total = len(assignments)
        if actual_total != expected_total:
            logger.error(f"Assignment count mismatch: {actual_total} vs expected {expected_total}")
            return False
        
        # Check that each image appears exactly once
        image_counts = {}
        for assignment in assignments:
            image_path = assignment.image_path
            image_counts[image_path] = image_counts.get(image_path, 0) + 1
        
        duplicate_images = [img for img, count in image_counts.items() if count > 1]
        if duplicate_images:
            logger.error(f"Found duplicate image assignments: {len(duplicate_images)} images")
            return False
        
        missing_images = expected_total - len(image_counts)
        if missing_images > 0:
            logger.error(f"Missing assignments for {missing_images} images")
            return False
        
        logger.info("Assignment validation passed")
        return True


def main():
    """Test the prompt distributor."""
    # Test configuration
    distribution = {
        'scene_understanding': 25,
        'observational_analytical': 25,
        'atmospheric_sensory': 25,
        'comprehensive_analysis': 25
    }
    
    # Create test image paths
    test_images = [f"test_image_{i:03d}.jpg" for i in range(100)]
    
    # Initialize distributor
    distributor = PromptDistributor(distribution, random_seed=42)
    
    # Distribute prompts
    assignments = distributor.distribute_prompts(test_images)
    
    # Validate
    is_valid = distributor.validate_assignments(assignments, len(test_images))
    print(f"Validation passed: {is_valid}")
    
    # Print summary
    summary = distributor.get_distribution_summary(assignments)
    print("\nActual distribution:")
    for category, count in summary.items():
        percentage = count / len(assignments) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Print some sample assignments
    print("\nSample assignments:")
    for i, assignment in enumerate(assignments[:5]):
        print(f"  {assignment.image_path}: {assignment.prompt_category}")
        print(f"    Prompt: {assignment.prompt[:60]}...")


if __name__ == "__main__":
    main()

