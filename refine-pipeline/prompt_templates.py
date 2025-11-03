"""
Human-like prompt templates for comprehensive visual understanding during pretraining.
These prompts are designed to extract diverse aspects of visual information.
"""

import random
from typing import List

class PromptTemplates:
    """Collection of diverse human-like prompts for visual understanding."""
    
    # Scene Understanding & Context
    SCENE_UNDERSTANDING = [
        "Tell me what's happening in this image and what the overall scene looks like.",
        "Walk me through what you see here, including the setting and any activities taking place.",
        "Describe the environment shown in this photo and what kind of place this appears to be.",
        "Explain what this image depicts, focusing on both the main subjects and their surroundings."
    ]
    
    # Observational & Analytical
    OBSERVATIONAL_ANALYTICAL = [
        "What details stand out to you most in this image, and how are different elements arranged?",
        "Take a close look at this photo and describe the key things you notice about the composition.",
        "Examine this image carefully and tell me about the various objects, people, or features you can identify.",
        "What catches your eye in this scene, and how would you characterize the overall visual elements?"
    ]
    
    # Atmospheric & Sensory
    ATMOSPHERIC_SENSORY = [
        "Describe the mood and atmosphere of this image, along with what creates that feeling.",
        "What kind of lighting, weather, or environmental conditions do you observe in this photo?",
        "Tell me about the colors, textures, and visual qualities that define this scene.",
        "How would you describe the ambiance and sensory aspects conveyed by this image?"
    ]
    
    # Narrative & Contextual
    NARRATIVE_CONTEXTUAL = [
        "What story does this image tell, and what clues help you understand the situation?",
        "Based on what you see, what do you think was happening before or might happen after this moment?",
        "Describe this scene as if you're explaining it to someone who can't see the image.",
        "What can you infer about the context, purpose, or significance of what's shown here?"
    ]
    
    # Spatial & Relational
    SPATIAL_RELATIONAL = [
        "How are the different elements in this image positioned relative to each other?",
        "Describe the layout and spatial relationships between the main components of this scene.",
        "What's in the foreground, middle ground, and background, and how do they relate?",
        "Explain the physical arrangement and organization of elements within this photo."
    ]
    
    # Functional & Purpose-Driven
    FUNCTIONAL_PURPOSE = [
        "What activities, functions, or purposes are suggested by what you see in this image?",
        "Based on the objects and setting, what might this location be used for?",
        "What can you tell me about the practical aspects or utility of what's shown here?",
        "How do the visible elements work together or serve specific functions in this context?"
    ]
    
    # Comparative & Detailed
    COMPARATIVE_DETAILED = [
        "Compare and contrast the different elements you see, noting their sizes, positions, and characteristics.",
        "What variations in color, shape, material, or other properties do you observe across the scene?",
        "How do the various components of this image differ from or complement each other?",
        "Describe both the similarities and differences among the objects or features present."
    ]
    
    # Perspective & Experience
    PERSPECTIVE_EXPERIENCE = [
        "If you were physically present in this scene, what would you likely notice, hear, or experience?",
        "Describe this image from the viewpoint of someone actually there observing the situation.",
        "What would it be like to be in this environment, and what details support that impression?",
        "How might different people experience or interpret this scene differently?"
    ]
    
    # Comprehensive Analysis
    COMPREHENSIVE_ANALYSIS = [
        "Provide a thorough analysis of this image, covering both obvious and subtle details.",
        "Break down everything visible in this photo, from major elements to smaller supporting details.",
        "Give me a complete overview of this scene, including both factual observations and interpretive insights.",
        "Analyze this image comprehensively, addressing visual elements, context, and any implications."
    ]
    
    @classmethod
    def get_all_categories(cls) -> dict:
        """Get all prompt categories as a dictionary."""
        return {
            'scene_understanding': cls.SCENE_UNDERSTANDING,
            'observational_analytical': cls.OBSERVATIONAL_ANALYTICAL,
            'atmospheric_sensory': cls.ATMOSPHERIC_SENSORY,
            'narrative_contextual': cls.NARRATIVE_CONTEXTUAL,
            'spatial_relational': cls.SPATIAL_RELATIONAL,
            'functional_purpose': cls.FUNCTIONAL_PURPOSE,
            'comparative_detailed': cls.COMPARATIVE_DETAILED,
            'perspective_experience': cls.PERSPECTIVE_EXPERIENCE,
            'comprehensive_analysis': cls.COMPREHENSIVE_ANALYSIS
        }
    
    @classmethod
    def get_all_prompts(cls) -> List[str]:
        """Get all prompts as a flat list."""
        all_prompts = []
        for category_prompts in cls.get_all_categories().values():
            all_prompts.extend(category_prompts)
        return all_prompts
    
    @classmethod
    def get_random_prompt_from_category(cls, category: str) -> str:
        """Get a random prompt from a specific category."""
        categories = cls.get_all_categories()
        if category not in categories:
            raise ValueError(f"Unknown category: {category}")
        return random.choice(categories[category])
    
    @classmethod
    def get_random_prompt(cls) -> str:
        """Get a random prompt from any category."""
        return random.choice(cls.get_all_prompts())
    
    @classmethod
    def get_balanced_prompts(cls, num_prompts: int) -> List[str]:
        """Get a balanced selection of prompts across all categories."""
        categories = cls.get_all_categories()
        prompts = []
        
        # Calculate how many prompts per category
        num_categories = len(categories)
        prompts_per_category = num_prompts // num_categories
        remaining = num_prompts % num_categories
        
        for i, (category, category_prompts) in enumerate(categories.items()):
            # Add extra prompt to first few categories if there's a remainder
            count = prompts_per_category + (1 if i < remaining else 0)
            
            # If we need more prompts than available in category, cycle through them
            for j in range(count):
                prompts.append(category_prompts[j % len(category_prompts)])
        
        # Shuffle to avoid predictable ordering
        random.shuffle(prompts)
        return prompts
