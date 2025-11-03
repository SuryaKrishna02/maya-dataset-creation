"""
Data processor for combining metadata from multiple JSONL files.
Handles depth_maps_metadata.jsonl, obj_extr_from_desc.jsonl, and fg_anno.jsonl.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImageMetadata:
    """Combined metadata for a single image."""
    image_path: str
    depth_metadata: Optional[str] = None
    extracted_objects_description: Optional[List[str]] = None
    scene_description: Optional[str] = None
    extracted_objects_image: Optional[List[str]] = None
    bounding_boxes: Optional[List[List[int]]] = None
    object_depths: Optional[List[int]] = None
    object_sizes: Optional[List[int]] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    def to_metadata_string(self) -> str:
        """Convert the metadata to a comprehensive system prompt using the provided template."""
        
        # Start with the introductory instruction
        prompt_parts = [
            "You are provided with rich visual and spatial metadata to generate comprehensive, human-like descriptions that capture both obvious and subtle aspects of the image. Use this information to provide detailed responses that address spatial relationships, atmospheric qualities, contextual meaning, and experiential elements."
        ]
        
        # Add Spatial and Depth Analysis section
        if self.depth_metadata:
            prompt_parts.append("\n**Spatial and Depth Analysis:**")
            prompt_parts.append(self.depth_metadata)
        
        # Add Object Detection and Localization section
        if (self.extracted_objects_image and self.bounding_boxes and 
            self.object_depths and self.object_sizes and 
            self.image_width is not None and self.image_height is not None):
            
            prompt_parts.append("\n**Object Detection and Localization:**")
            prompt_parts.append(f"For image dimensions of height={self.image_height} and width={self.image_width} pixels:")
            prompt_parts.append("Descriptive Objects | Bounding Box Coordinates | Object Depth | Area")
            
            # Create table rows for each detected object
            for i, (obj_desc, bbox, depth, size) in enumerate(zip(
                self.extracted_objects_image, 
                self.bounding_boxes, 
                self.object_depths, 
                self.object_sizes
            )):
                bbox_str = f"[{', '.join(map(str, bbox))}]"
                prompt_parts.append(f'"{obj_desc}" | {bbox_str} | {depth} | {size}')
        
        # Add Scene Context and Elements section
        if self.extracted_objects_description or self.scene_description:
            prompt_parts.append("\n**Scene Context and Elements:**")
            
            if self.extracted_objects_description:
                objects_list = '", "'.join(self.extracted_objects_description)
                prompt_parts.append(f'Objects: "{objects_list}"')
            
            if self.scene_description:
                prompt_parts.append(f"Concise Description: {self.scene_description}")
        
        # Add Response Guidelines section
        prompt_parts.append("\n**Response Guidelines:**")
        prompt_parts.append(
            "When generating responses, integrate this metadata to provide rich descriptions that include: "
            "spatial relationships and depth perception using the depth map data; "
            "detailed object analysis incorporating size, position, and distance information relative to the "
            f"{self.image_height}x{self.image_width} pixel image dimensions; "
            "text recognition and OCR content when present (denoted as \"<OCR_TEXT>\"); "
            "atmospheric and environmental context suggested by the scene composition; "
            "comparative analysis of elements at different depths and positions; "
            "interpretive insights about the setting, purpose, and potential narrative; "
            "experiential descriptions that consider what it would be like to be present in the scene; "
            "comprehensive coverage of both prominent features and subtle supporting details; "
            "contextual understanding that connects individual elements into a cohesive scene interpretation. "
            "Use natural, flowing paragraph format without markdown formatting, ensuring responses are "
            "thorough enough to help train comprehensive visual understanding while remaining engaging and human-readable."
        )
        
        return "\n".join(prompt_parts)


class DataProcessor:
    """Processes and combines metadata from multiple JSONL files."""
    
    def __init__(self, data_dir: str, metadata_dir: str = "metadata"):
        """Initialize the data processor with the directory containing JSONL files."""
        self.data_dir = Path(data_dir)
        self.metadata_dir = self.data_dir / metadata_dir
        self.depth_file = self.metadata_dir / "depth_maps_metadata.jsonl"
        self.obj_desc_file = self.metadata_dir / "obj_extr_from_desc.jsonl"
        self.fg_anno_file = self.metadata_dir / "fg_anno.jsonl"
        
        # Validate files exist
        self._validate_files()
        
    def _validate_files(self) -> None:
        """Validate that all required files exist."""
        files = [self.depth_file, self.obj_desc_file, self.fg_anno_file]
        missing_files = [f for f in files if not f.exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")
        
        logger.info(f"Found all required files in {self.metadata_dir}")
    
    def _normalize_image_path(self, image_path: str) -> str:
        """Normalize image path for consistent matching across files."""
        # Remove 'extracted_images/' prefix if present
        if image_path.startswith('extracted_images/'):
            return image_path[len('extracted_images/'):]
        return image_path
    
    def _load_jsonl_file(self, file_path: Path) -> Dict[str, dict]:
        """Load a JSONL file and return a dictionary keyed by normalized image path."""
        data = {}
        line_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        if 'image' in item:
                            normalized_path = self._normalize_image_path(item['image'])
                            data[normalized_path] = item
                            line_count += 1
                        else:
                            logger.warning(f"No 'image' field in line {line_num} of {file_path}")
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in {file_path} line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
        
        logger.info(f"Loaded {line_count} items from {file_path}")
        return data
    
    def load_all_data(self) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
        """Load all three JSONL files."""
        logger.info("Loading all JSONL files...")
        
        depth_data = self._load_jsonl_file(self.depth_file)
        obj_desc_data = self._load_jsonl_file(self.obj_desc_file)
        fg_anno_data = self._load_jsonl_file(self.fg_anno_file)
        
        return depth_data, obj_desc_data, fg_anno_data
    
    def combine_metadata(self, image_path: str, depth_data: dict, obj_desc_data: dict, fg_anno_data: dict) -> ImageMetadata:
        """Combine metadata from all sources for a single image."""
        normalized_path = self._normalize_image_path(image_path)
        
        # Get data from each source
        depth_info = depth_data.get(normalized_path, {})
        obj_desc_info = obj_desc_data.get(normalized_path, {})
        fg_anno_info = fg_anno_data.get(normalized_path, {})
        
        # Create combined metadata
        metadata = ImageMetadata(
            image_path=image_path,
            depth_metadata=depth_info.get('depth_metadata'),
            extracted_objects_description=obj_desc_info.get('extr_obj_fr_desc'),
            scene_description=obj_desc_info.get('description'),
            extracted_objects_image=fg_anno_info.get('extr_obj_from_img'),
            bounding_boxes=fg_anno_info.get('bounding_boxes'),
            object_depths=fg_anno_info.get('object_depth'),
            object_sizes=fg_anno_info.get('size'),
            image_width=fg_anno_info.get('width'),
            image_height=fg_anno_info.get('height')
        )
        
        return metadata
    
    def get_all_image_paths(self, require_all_metadata: bool = True) -> List[str]:
        """Get image paths from all three files.
        
        Args:
            require_all_metadata: If True, only return images present in ALL three files.
                                If False, return images from any file.
        """
        depth_data, obj_desc_data, fg_anno_data = self.load_all_data()
        
        if require_all_metadata:
            # Only images present in ALL three files
            depth_paths = set(depth_data.keys())
            obj_desc_paths = set(obj_desc_data.keys())
            fg_anno_paths = set(fg_anno_data.keys())
            
            common_paths = depth_paths.intersection(obj_desc_paths).intersection(fg_anno_paths)
            logger.info(f"Found {len(common_paths)} images present in all three metadata files")
            logger.info(f"  Depth metadata: {len(depth_paths)} images")
            logger.info(f"  Object descriptions: {len(obj_desc_paths)} images") 
            logger.info(f"  Foreground annotations: {len(fg_anno_paths)} images")
            
            return sorted(common_paths)
        else:
            # All unique images from any file
            all_paths = set()
            all_paths.update(depth_data.keys())
            all_paths.update(obj_desc_data.keys())
            all_paths.update(fg_anno_data.keys())
            
            logger.info(f"Found {len(all_paths)} total unique images across all files")
            return sorted(all_paths)
    
    def process_batch(self, image_paths: List[str], batch_size: int = 1000) -> Generator[List[ImageMetadata], None, None]:
        """Process images in batches to manage memory usage."""
        logger.info(f"Processing {len(image_paths)} images in batches of {batch_size}")
        
        # Load all data once
        depth_data, obj_desc_data, fg_anno_data = self.load_all_data()
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_metadata = []
            
            for image_path in batch_paths:
                metadata = self.combine_metadata(image_path, depth_data, obj_desc_data, fg_anno_data)
                batch_metadata.append(metadata)
            
            logger.info(f"Processed batch {i // batch_size + 1}, images {i + 1}-{min(i + batch_size, len(image_paths))}")
            yield batch_metadata
    
    def get_metadata_for_image(self, image_path: str) -> ImageMetadata:
        """Get combined metadata for a single image."""
        depth_data, obj_desc_data, fg_anno_data = self.load_all_data()
        return self.combine_metadata(image_path, depth_data, obj_desc_data, fg_anno_data)
    
    def get_sample_data(self, num_samples: int = 10) -> List[ImageMetadata]:
        """Get a sample of combined metadata for testing."""
        all_paths = self.get_all_image_paths()
        sample_paths = all_paths[:num_samples]
        
        depth_data, obj_desc_data, fg_anno_data = self.load_all_data()
        
        sample_metadata = []
        for image_path in sample_paths:
            metadata = self.combine_metadata(image_path, depth_data, obj_desc_data, fg_anno_data)
            sample_metadata.append(metadata)
        
        return sample_metadata


def main():
    """Test the data processor."""
    processor = DataProcessor(".")
    
    # Test with filtering for common images
    common_images = processor.get_all_image_paths(require_all_metadata=True)
    print(f"Images present in all three files: {len(common_images)}")
    
    # Get sample data
    sample_data = processor.get_sample_data(3)
    
    for i, metadata in enumerate(sample_data):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Image: {metadata.image_path}")
        print(f"Has depth: {metadata.depth_metadata is not None}")
        print(f"Has objects from desc: {metadata.extracted_objects_description is not None}")
        print(f"Has scene desc: {metadata.scene_description is not None}")
        print(f"Has objects from img: {metadata.extracted_objects_image is not None}")
        print("\nMetadata string:")
        print(metadata.to_metadata_string())


if __name__ == "__main__":
    main()
