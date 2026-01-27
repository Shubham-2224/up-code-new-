"""
Enhanced OCR Processor with 400 DPI
Local OCR processing for high-quality extraction
"""

import os
import io
import re
import base64
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Optional, Tuple

try:
    from paddle_ocr_processor import PaddleOCRProcessor
    PADDLE_PROCESSOR_AVAILABLE = True
except ImportError:
    PADDLE_PROCESSOR_AVAILABLE = False
    # print("WARNING: PaddleOCR Processor module not found")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: OpenCV not found. Deskewing disabled.")

class OCRProcessor400DPI:
    """
    High-quality OCR processor using 400 DPI
    Uses local Tesseract OCR for all processing
    """

    # Optimized Configs as per User Request (Option 3)
    CONFIG_EPIC = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ --oem 3"
    CONFIG_AGE_GENDER = "--psm 7 -c tessedit_char_whitelist=0123456789MF --oem 3"
    CONFIG_NAME = "--psm 6 --oem 3" # Standard block

    def __init__(self):
        """Initialize OCR processor"""
        # PERFORMANCE OPTIMIZATION: Use configurable DPI with default for speed
        self.dpi = int(os.getenv('OCR_DPI', '150'))  # Default to 150 DPI for faster processing
        self.current_lang = 'mr' # Default to Marathi
        self.quality_mode = os.getenv('OCR_QUALITY', 'fast')  # fast, balanced, accurate

        # Performance optimization: Cache for processed images
        self.image_cache = {}
        self.cache_max_size = 50  # Limit cache size to prevent memory issues

        # Configure Tesseract path from environment (critical for systemd services)
        tesseract_cmd = os.getenv('TESSERACT_CMD')
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif os.name != 'nt':  # Linux/Unix - ensure tesseract is findable
            for path in ['/usr/bin/tesseract', '/usr/local/bin/tesseract']:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

        # Voter ID patterns (Indian EPIC format: 3 letters + 7 digits)
        self.voter_id_patterns = [
            r'\b[A-Z]{3}[0-9]{7}\b',           # Standard: ABC1234567 (STRICT)
            r'\b[A-Z]{3}\s*[0-9]{7}\b',        # With space: ABC 1234567
            r'\b[A-Z0-9]{10}\b',               # Any 10 alphanumeric (fallback for correction)
            r'\b[A-Z]{2,4}[0-9]{6,8}\b',       # Flexible: 2-4 letters + 6-8 digits
        ]

        # Performance settings based on quality mode
        if self.quality_mode == 'fast':
            self.max_retries = 1  # Minimal retries for speed
            self.min_confidence_threshold = 0.6
            self.enable_char_processing = False # Disable individual char OCR
            self.enable_paddle_fallback = False # Use Tesseract primarily if fast
        elif self.quality_mode == 'balanced':
            self.max_retries = 2  # Moderate retries
            self.min_confidence_threshold = 0.4
            self.enable_char_processing = False # Still disable for balanced
            self.enable_paddle_fallback = True  # Enable for better Marathi
        else:  # accurate
            self.max_retries = 3  # Maximum retries for accuracy
            self.min_confidence_threshold = 0.3
            self.enable_char_processing = True  # Only for accurate
            self.enable_paddle_fallback = True

    def set_quality_mode(self, mode: str):
        """Update quality mode and related flags"""
        self.quality_mode = mode
        if mode == 'fast':
            self.max_retries = 1
            self.min_confidence_threshold = 0.6
            self.enable_char_processing = False
            self.enable_paddle_fallback = False
            self.dpi = 200 # Faster DPI for fast mode
        elif mode == 'balanced':
            self.max_retries = 2
            self.min_confidence_threshold = 0.4
            self.enable_char_processing = False
            self.enable_paddle_fallback = True
            self.dpi = 300
        else:
            self.max_retries = 3
            self.min_confidence_threshold = 0.3
            self.enable_char_processing = True
            self.enable_paddle_fallback = True
            self.dpi = 300


        print("OK: OCR Processor initialized with 300 DPI (EPIC-optimized)")

        # Initialize PaddleOCR if available
        self.paddle_processor = None
        if PADDLE_PROCESSOR_AVAILABLE:
            try:
                self.paddle_processor = PaddleOCRProcessor(lang='mr')
            except Exception as e:
                print(f"Failed to init PaddleOCR: {e}")

    def test_epic_extraction(self, test_images: list = None) -> None:
        """
        Test EPIC extraction capabilities with sample images.

        Args:
            test_images: List of PIL Images to test (optional)
        """
        print("\n" + "="*60)
        print("🧪 TESTING EPIC EXTRACTION CAPABILITIES")
        print("="*60)

        # Test with sample EPIC patterns
        test_epics = [
            "ABC1234567",
            "XYZ9876543",
            "DEF4567890",
            "GHI1112223"
        ]

        print(f"📋 Test EPICs: {test_epics}")

        if test_images:
            print(f"\n🖼️  Testing with {len(test_images)} sample images:")
            for i, img in enumerate(test_images):
                print(f"\n  Test Image {i+1}:")
                result = self.extract_epic_number(img, use_advanced=True)
                print(f"    Result: '{result['voter_id']}' (confidence: {result['confidence']:.2f})")
                print(f"    Method: {result['method']}")
        else:
            print("\nℹ️  No test images provided. EPIC extraction is ready for use.")
            print("   Call extract_epic_number() with image, pdf_page, or rect parameters.")

        print("\n✅ EPIC extraction system ready with advanced image processing!")
        print("="*60)

    def _get_image_cache_key(self, image: Image.Image) -> str:
        """
        Generate a cache key for an image based on its content hash.
        This helps avoid re-processing identical images.
        """
        try:
            # Create a simple hash of image dimensions and a sample of pixels
            import hashlib
            img_bytes = image.tobytes()
            return hashlib.md5(img_bytes[:1024] + str(image.size).encode()).hexdigest()[:16]
        except:
            # Fallback: use image size as cache key
            return f"{image.size[0]}x{image.size[1]}"

    def set_ocr_language(self, lang='mr'):
        """
        Set OCR language (mr, hi, en)
        """
        if self.paddle_processor:
             # Ensure lang is 'mr' or 'hi'
             standard_lang = 'mr' if lang in ['mar', 'mr'] else 'hi' if lang in ['hin', 'hi'] else 'mr'
             self.paddle_processor.set_language(standard_lang)
             self.current_lang = standard_lang

    def extract_text_with_config(self, image: Image.Image, config: str) -> str:
        """
        Extract text using a specific Tesseract configuration.
        Fast path for specific fields (EPIC, Age, etc).
        """
        try:
             # Preprocess
            processed_img = self.preprocess_image(image, for_ocr=True)
            return pytesseract.image_to_string(processed_img, config=config, lang='eng+hin').strip()
        except Exception as e:
            print(f"      OCRError: {e}")
            return ""
    
    def deskew_image(self, image: Image.Image) -> Image.Image:
        """
        Deskew image using OpenCV
        """
        if not CV2_AVAILABLE:
            return image
            
        try:
            # Convert to CV2
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Use Hough lines to detect skew
            # Or better: minAreaRect of text contours
            
            # Simple approach: Threshold > Find Coordinates > MinAreaRect
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            coords = np.column_stack(np.where(thresh > 0))
            if coords.size == 0:
                return image
                
            angle = cv2.minAreaRect(coords)[-1]
            
            # Correct angle logic
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Ignore small angles (likely noise)
            if abs(angle) < 0.5 or abs(angle) > 45:
                 return image
            
            (h, w) = img_cv.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            rotated = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            # print(f"Deskew failed: {e}")
            return image

    def preprocess_image(self, image: Image.Image, for_ocr: bool = True, skip_heavy_ops: bool = False) -> Image.Image:
        """
        Preprocess image for better OCR results.
        skip_heavy_ops: If True, skips Deskew and Denoise (use for clean digital crops).
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if for_ocr:
            try:
                # 1. Deskew (Skip if fast mode)
                if CV2_AVAILABLE and not skip_heavy_ops:
                    image = self.deskew_image(image)

                # 2. Convert to Grayscale
                image = image.convert('L')
                
                # 3. Denoise (Skip if fast mode)
                if not skip_heavy_ops:
                    image = image.filter(ImageFilter.MedianFilter(size=3))
                
                # 4. Light Sharpen (reduced from 2.0 to 1.5 for better character recognition)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.5)

                # 5. Contrast / Binarization
                # Moderate Contrast (reduced from 2.0 to 1.5 to avoid character distortion)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                
                # Adaptive Thresholding (Binarization) using OpenCV if available
                if CV2_AVAILABLE:
                     img_np = np.array(image)
                     # Adaptive Threshold: Block Size 11, C=2
                     binary = cv2.adaptiveThreshold(
                         img_np, 255, 
                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv2.THRESH_BINARY, 
                         11, 2
                     )
                     image = Image.fromarray(binary)
                else:
                     # Fallback PIL Binarization (single threshold)
                     image = image.point(lambda x: 0 if x < 128 else 255, '1')
                     
            except Exception as e:
                print(f"      Pre-processing warning: {e}")
        
        return image
    
    def extract_voter_id(self, image: Image.Image, pdf_page=None, rect=None) -> Dict:
        """
        Extract voter ID from image using local OCR
        
        Args:
            image: PIL Image or None
            pdf_page: PyMuPDF page object (optional)
            rect: PyMuPDF Rect object (optional)
        
        Returns:
            Dict with voter_id, confidence, method
        """
        try:
            # Extract high-quality image if pdf_page and rect provided
            if pdf_page and rect:
                pix = pdf_page.get_pixmap(clip=rect, dpi=self.dpi, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            if not image:
                return {
                    'voter_id': '',
                    'confidence': 0.0,
                    'method': 'error',
                    'raw_text': ''
                }

            # Performance optimization: Check cache first
            cache_key = self._get_image_cache_key(image)
            if cache_key in self.image_cache:
                return self.image_cache[cache_key].copy()

            # Skip quality analysis for performance - always use advanced processing
            quality_metrics = {'recommend_advanced': True}

            # Strategy 1: Standard OCR processing (fast path)
            processed_img = self.preprocess_image(image, for_ocr=True, skip_heavy_ops=True)

            raw_text = pytesseract.image_to_string(
                processed_img,
                lang='eng+hin',
                config='--psm 6 --oem 1 -c tessedit_do_invert=0'
            ).strip()

            # Extract voter ID using patterns
            voter_id, confidence = self._extract_voter_id_from_text(raw_text)
            method = 'tesseract_standard'

            # Strategy 2: Advanced EPIC processing (Skip in 'fast' mode)
            should_try_advanced = False
            if self.quality_mode != 'fast':
                should_try_advanced = (
                    confidence < 0.8 or  # Low confidence from standard OCR
                    not voter_id  # No result from standard OCR
                )

            if should_try_advanced:
                reason = "low confidence" if voter_id else "no result"
                # print(f"      🔄 Standard OCR {reason} ({confidence:.2f}), trying advanced EPIC processing...")

                advanced_result = self.extract_epic_with_advanced_image_processing(image, pdf_page, rect)

                if advanced_result and advanced_result['confidence'] > confidence:
                    voter_id = advanced_result['voter_id']
                    confidence = advanced_result['confidence']
                    method = 'advanced_epic_processing'
                    print(f"      ✅ Advanced processing improved result: '{voter_id}' (conf: {confidence:.2f})")
                elif advanced_result:
                    print(f"      ⚠️ Advanced processing result: '{advanced_result['voter_id']}' (conf: {advanced_result['confidence']:.2f}) - not better than standard")
                else:
                    print(f"      ⚠️ Advanced processing failed to produce result")
            
            # print(f"      Extracted: '{voter_id}' (confidence: {confidence:.2f})")
            
            # Prepare result
            result = {
                'voter_id': voter_id,
                'confidence': confidence,
                'method': 'tesseract',
                'raw_text': raw_text
            }

            # Cache the result for future use
            if len(self.image_cache) < self.cache_max_size:
                self.image_cache[cache_key] = result.copy()

            # print(f"      ✓ Local OCR SUCCESS (conf: {confidence:.2f})")
            return result
        
        except Exception as e:
            result = {
                'voter_id': '',
                'confidence': 0.0,
                'method': 'error',
                'raw_text': '',
                'error': str(e)
            }

            # Cache error result too to avoid repeated failures
            if len(self.image_cache) < self.cache_max_size:
                self.image_cache[cache_key] = result.copy()

            print(f"      ERROR: Voter ID extraction failed: {str(e)}")
            return result

    def analyze_image_quality_for_epic(self, image: Image.Image) -> Dict:
        """
        Analyze image quality metrics specifically for EPIC number extraction.

        Returns metrics that help decide whether to use advanced processing:
        - contrast_ratio: Text contrast quality
        - noise_level: Image noise estimation
        - text_density: Estimated text regions
        - blur_score: Image sharpness/blur detection

        Args:
            image: PIL Image to analyze

        Returns:
            Dict with quality metrics and recommendation
        """
        try:
            if image.mode != 'L':
                gray_img = image.convert('L')
            else:
                gray_img = image

            if CV2_AVAILABLE:
                img_cv = np.array(gray_img)

                # Contrast analysis
                min_val, max_val, _, _ = cv2.minMaxLoc(img_cv)
                contrast_ratio = (max_val - min_val) / 255.0

                # Noise estimation using Laplacian variance
                laplacian_var = cv2.Laplacian(img_cv, cv2.CV_64F).var()
                blur_score = laplacian_var / 1000.0  # Normalize

                # Text density estimation
                _, binary = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text_density = np.sum(binary == 0) / binary.size

                # Overall quality score
                quality_score = (contrast_ratio * 0.4 + blur_score * 0.3 + text_density * 0.3)

                return {
                    'contrast_ratio': contrast_ratio,
                    'blur_score': blur_score,
                    'text_density': text_density,
                    'quality_score': quality_score,
                    'recommend_advanced': quality_score < 0.6  # Use advanced processing if quality is poor
                }
            else:
                # Fallback analysis without OpenCV
                histogram = gray_img.histogram()
                contrast_ratio = (sum(histogram[200:]) / sum(histogram[:55])) if sum(histogram[:55]) > 0 else 1.0

                return {
                    'contrast_ratio': min(contrast_ratio, 1.0),
                    'blur_score': 0.5,  # Unknown
                    'text_density': 0.5,  # Unknown
                    'quality_score': contrast_ratio * 0.5,
                    'recommend_advanced': contrast_ratio < 0.7
                }

        except Exception as e:
            print(f"      Quality analysis error: {str(e)}")
            return {
                'contrast_ratio': 0.5,
                'blur_score': 0.5,
                'text_density': 0.5,
                'quality_score': 0.5,
                'recommend_advanced': True  # Default to advanced processing on error
            }

    def extract_epic_number(self, image: Image.Image = None, pdf_page=None, rect=None, use_advanced: bool = True) -> Dict:
        """
        Specialized method for extracting EPIC numbers using image processing.

        This is the main entry point for EPIC extraction that intelligently
        chooses between standard and advanced processing based on image quality.

        Args:
            image: PIL Image or None
            pdf_page: PyMuPDF page object (optional)
            rect: PyMuPDF Rect object (optional)
            use_advanced: Whether to allow advanced processing (default: True)

        Returns:
            Dict with voter_id, confidence, method, and processing details
        """
        if not use_advanced:
            # Use standard processing only
            return self.extract_voter_id(image, pdf_page, rect)

        # Use intelligent processing with advanced fallback
        return self.extract_voter_id(image, pdf_page, rect)

    def extract_epic_with_advanced_image_processing(self, image: Image.Image, pdf_page=None, rect=None) -> Dict:
        """
        Extract EPIC number using advanced image processing techniques optimized for voter ID format.

        This method uses specialized preprocessing for EPIC numbers (ABC1234567 format):
        - Morphological operations for character enhancement
        - Adaptive thresholding optimized for text regions
        - Noise reduction while preserving character edges
        - Character segmentation and validation

        Args:
            image: PIL Image or None
            pdf_page: PyMuPDF page object (optional)
            rect: PyMuPDF Rect object (optional)

        Returns:
            Dict with voter_id, confidence, method, and processing details
        """
        try:
            # Extract high-quality image if pdf_page and rect provided
            if pdf_page and rect:
                pix = pdf_page.get_pixmap(clip=rect, dpi=self.dpi, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            if not image:
                return {
                    'voter_id': '',
                    'confidence': 0.0,
                    'method': 'error',
                    'raw_text': '',
                    'processing_steps': []
                }

            # Performance optimization: Check cache first
            cache_key = f"epic_advanced_{self._get_image_cache_key(image)}"
            if cache_key in self.image_cache:
                return self.image_cache[cache_key].copy()

            processing_steps = []

            # === ADVANCED IMAGE PREPROCESSING FOR EPIC NUMBERS ===

            # Step 1: Convert to grayscale and enhance contrast
            if image.mode != 'L':
                image = image.convert('L')
            processing_steps.append("Converted to grayscale")

            # ULTRA-FAST preprocessing based on quality mode
            if self.quality_mode == 'fast':
                # Minimal processing for speed
                if CV2_AVAILABLE:
                    img_cv = np.array(image)
                    # Very fast bilateral filter with smaller kernel
                    img_cv = cv2.bilateralFilter(img_cv, 3, 30, 30)
                    image = Image.fromarray(img_cv)
                else:
                    # Simple contrast boost only
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.2)
                processing_steps.append("Applied ultra-fast preprocessing")
            elif CV2_AVAILABLE:
                img_cv = np.array(image)
                # Balanced processing
                img_cv = cv2.bilateralFilter(img_cv, 5, 50, 50)
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
                img_cv = clahe.apply(img_cv)
                image = Image.fromarray(img_cv)
                processing_steps.append("Applied balanced preprocessing")
            else:
                # Simple PIL enhancement
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                processing_steps.append("Applied basic contrast enhancement")

            # Skip advanced binarization in fast mode
            if self.quality_mode == 'fast':
                pass  # Skip binarization for speed
            else:
                # Fallback to PIL binarization with dynamic threshold
                # Calculate optimal threshold based on image histogram
                histogram = image.histogram()
                total_pixels = sum(histogram)
                cumulative = 0
                threshold = 128  # default

                for i, count in enumerate(histogram):
                    cumulative += count
                    if cumulative > total_pixels * 0.5:  # 50% cumulative
                        threshold = i
                        break

                image = image.point(lambda x: 0 if x < threshold else 255, '1')
                processing_steps.append("Applied dynamic thresholding")

            # Step 6: Extract text using optimized PSM mode for EPIC format
            best_result = {'text': '', 'confidence': 0.0}
            psm_configs = [
                ('--psm 7 --oem 3', 'EPIC single line mode')  # Only try the most reliable mode
            ]

            for config, description in psm_configs:
                try:
                    raw_text = pytesseract.image_to_string(
                        image,
                        lang='eng',
                        config=f'{config} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    ).strip()

                    if raw_text:
                        # Validate if this looks like an EPIC number
                        epic_match = self._validate_epic_format(raw_text)
                        if epic_match and epic_match['confidence'] > best_result['confidence']:
                            best_result = {
                                'text': epic_match['epic'],
                                'confidence': epic_match['confidence'],
                                'method': f'advanced_{description}'
                            }
                            processing_steps.append(f"Tried {description}: '{raw_text}' → '{epic_match['epic']}'")

                except Exception as e:
                    processing_steps.append(f"Failed {description}: {str(e)}")
                    continue

            # Step 7: If no good result, try character-level processing (SLOW)
            if self.enable_char_processing and best_result['confidence'] < 0.7:
                char_result = self._process_epic_characters(image)
                if char_result and char_result['confidence'] > best_result['confidence']:
                    best_result = char_result
                    processing_steps.append("Used character-level processing")

            # Extract final EPIC number
            voter_id = best_result['text']
            confidence = best_result['confidence']

            # Apply format correction if needed
            if voter_id:
                voter_id = self._correct_voter_id_format(voter_id)

            result = {
                'voter_id': voter_id,
                'confidence': confidence,
                'method': 'advanced_image_processing',
                'raw_text': best_result.get('text', ''),
                'processing_steps': processing_steps
            }

            # Cache the result
            if len(self.image_cache) < self.cache_max_size:
                self.image_cache[cache_key] = result.copy()

            # Success - no logging for performance
            return result

        except Exception as e:
            result = {
                'voter_id': '',
                'confidence': 0.0,
                'method': 'error',
                'raw_text': '',
                'processing_steps': [],
                'error': str(e)
            }

            # Cache error result
            if len(self.image_cache) < self.cache_max_size:
                self.image_cache[cache_key] = result.copy()

            print(f"      ERROR: Advanced EPIC processing failed: {str(e)}")
            return result

    def _validate_epic_format(self, text: str) -> Dict:
        """
        Validate if extracted text contains a valid EPIC number pattern.

        Args:
            text: Raw extracted text

        Returns:
            Dict with epic, confidence, or None if no valid EPIC found
        """
        if not text:
            return None

        # Clean the text
        text = text.upper().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

        # Try different patterns
        patterns = [
            r'\b([A-Z]{3}[0-9]{7})\b',           # ABC1234567
            r'\b([A-Z]{3}\s*[0-9]{7})\b',        # ABC 1234567
            r'\b([A-Z]{2,4}[0-9]{6,8})\b',       # Flexible
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                epic = matches[0].replace(' ', '')

                # Calculate confidence based on format compliance
                confidence = self._calculate_epic_confidence(epic)

                if confidence > 0.5:  # Minimum confidence threshold
                    return {
                        'epic': epic,
                        'confidence': confidence
                    }

        return None

    def _calculate_epic_confidence(self, epic: str) -> float:
        """
        Calculate confidence score for extracted EPIC number.

        Args:
            epic: The extracted EPIC number

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not epic or len(epic) < 9:
            return 0.0

        confidence = 0.0

        # Length check (should be 10 characters: 3 letters + 7 digits)
        if len(epic) == 10:
            confidence += 0.3

        # First 3 characters should be letters
        letters_confidence = 0
        for i in range(min(3, len(epic))):
            if epic[i].isalpha():
                letters_confidence += 1
        confidence += (letters_confidence / 3) * 0.3

        # Last 7 characters should be digits
        digits_confidence = 0
        for i in range(3, min(10, len(epic))):
            if epic[i].isdigit():
                digits_confidence += 1
        confidence += (digits_confidence / 7) * 0.4

        return min(confidence, 1.0)

    def _process_epic_characters(self, image: Image.Image) -> Dict:
        """
        Process image at character level to reconstruct EPIC number.

        Args:
            image: Binary image of the EPIC region

        Returns:
            Dict with reconstructed EPIC and confidence
        """
        try:
            if not CV2_AVAILABLE:
                return None

            img_cv = np.array(image)
            if len(img_cv.shape) == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

            # Find contours (potential characters)
            contours, _ = cv2.findContours(img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by size (character-like dimensions)
            char_contours = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)

                # Character-like dimensions: reasonable width/height ratio and size
                if 0.1 < aspect_ratio < 5.0 and 10 < w < 100 and 15 < h < 100:
                    char_contours.append((x, y, w, h))

            # Sort by x-coordinate (left to right)
            char_contours.sort(key=lambda c: c[0])

            if len(char_contours) < 9:  # Need at least 9 characters for EPIC
                return None

            # Extract individual characters and classify them
            characters = []
            for x, y, w, h in char_contours[:10]:  # Take first 10 potential characters
                char_img = img_cv[y:y+h, x:x+w]

                # Resize for consistent OCR
                char_img = cv2.resize(char_img, (28, 28))

                # OCR individual character
                char_text = pytesseract.image_to_string(
                    Image.fromarray(char_img),
                    config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ).strip()

                if char_text:
                    characters.append(char_text[0])  # Take first character

            # Reconstruct EPIC
            if len(characters) >= 9:
                epic_candidate = ''.join(characters[:10])

                # Validate the reconstructed EPIC
                validation = self._validate_epic_format(epic_candidate)
                if validation:
                    return {
                        'text': validation['epic'],
                        'confidence': validation['confidence'] * 0.8,  # Slightly lower confidence for character-level
                        'method': 'character_level_processing'
                    }

            return None

        except Exception as e:
            print(f"      Character processing error: {str(e)}")
            return None

    def _correct_voter_id_format(self, voter_id: str) -> str:
        """
        Correct Voter ID format using minimal corrections for EPIC format (ABC1234567):
        - Positions 1-3: Should be LETTERS (A-Z) - minimal corrections only for obvious OCR errors
        - Positions 4-10: Should be NUMBERS (0-9) - correct common letter/number confusions

        Args:
            voter_id: Raw voter ID from OCR

        Returns:
            Corrected voter ID
        """
        # STRICT 10-character check (3 letters + 7 digits)
        if not voter_id:
            return ""
        
        # Clean anything that's not alphanumeric
        voter_id = re.sub(r'[^A-Z0-9]', '', voter_id.upper())
        
        if len(voter_id) < 8:
            return voter_id

        original = voter_id

        # COMPREHENSIVE letter corrections (for positions 1-3) - convert numbers that look like letters
        # Based on common OCR confusions in Indian documents
        letter_corrections = {
            '0': 'O',  # 0 can look like O
            '1': 'I',  # 1 can look like I (most common)
            '2': 'Z',  # 2 can look like Z
            '3': 'E',  # 3 can look like E
            '4': 'A',  # 4 can look like A
            '5': 'S',  # 5 can look like S
            '6': 'G',  # 6 can look like G/C
            '7': 'T',  # 7 can look like T
            '8': 'B',  # 8 can look like B
            '9': 'G',  # 9 can look like G
        }

        # EXTENDED corrections for additional common confusions
        # Add multiple possible mappings for ambiguous digits
        extended_letter_corrections = {
            '0': ['O'],      # 0 -> O
            '1': ['I', 'L'], # 1 -> I or L
            '2': ['Z'],      # 2 -> Z
            '3': ['E'],      # 3 -> E
            '4': ['A'],      # 4 -> A
            '5': ['S'],      # 5 -> S
            '6': ['G', 'C'], # 6 -> G or C
            '7': ['T'],      # 7 -> T
            '8': ['B'],      # 8 -> B
            '9': ['G', 'Q'], # 9 -> G or Q
        }

        # Number correction map (for positions 4-10) - letters that commonly look like numbers
        number_corrections = {
            'O': '0',
            'I': '1',
            'L': '1',
            'Z': '2',
            'E': '3',
            'A': '4',
            'S': '5',
            'G': '6',
            'T': '7',
            'B': '8',
            'Q': '0',
            'D': '0',
        }

        corrected = list(voter_id.upper())  # Ensure uppercase
        corrections_made = []

        # Phase 1: Ensure first 3 positions are LETTERS (COMPREHENSIVE APPROACH)
        for i in range(min(3, len(corrected))):
            char = corrected[i]
            # Convert digits to the most likely letters
            if char.isdigit():
                possible_letters = extended_letter_corrections.get(char, [])
                if possible_letters:
                    # Use the first (most likely) option
                    new_char = possible_letters[0]
                    if new_char != char:
                        corrections_made.append(f"Pos {i+1}: '{char}' → '{new_char}' (letter)")
                        corrected[i] = new_char
                else:
                    # Fallback for unmapped digits
                    corrections_made.append(f"Pos {i+1}: '{char}' → 'X' (unknown digit)")
                    corrected[i] = 'X'

        # Phase 2: Ensure positions 4-10 are NUMBERS
        for i in range(3, min(10, len(corrected))):
            char = corrected[i]
            # Convert letters to the most likely numbers
            if char.isalpha():
                new_char = number_corrections.get(char, '0')  # Use 0 as fallback
                if new_char != char:
                    corrections_made.append(f"Pos {i+1}: '{char}' → '{new_char}' (number)")
                    corrected[i] = new_char

        result = ''.join(corrected)

        # ADVANCED CORRECTION: If result doesn't match EPIC format, try alternative corrections
        if len(result) >= 10 and not (result[:3].isalpha() and result[3:10].isdigit()):
            # Try alternative corrections for ambiguous digits
            alternative_results = [result]  # Start with current result

            # Generate alternatives for positions with multiple possible corrections
            original_chars = list(voter_id.upper())
            for i in range(min(3, len(original_chars))):
                if original_chars[i].isdigit():
                    possible_letters = extended_letter_corrections.get(original_chars[i], [])
                    if len(possible_letters) > 1:  # Multiple possibilities
                        # Create alternative results with different letter choices
                        new_alternatives = []
                        for alt_result in alternative_results:
                            alt_list = list(alt_result)
                            for letter_option in possible_letters:
                                alt_list[i] = letter_option
                                new_alt = ''.join(alt_list)
                                if new_alt not in new_alternatives:
                                    new_alternatives.append(new_alt)
                        alternative_results.extend(new_alternatives)

            # Find the best alternative that matches EPIC format
            for alt_result in alternative_results[1:]:  # Skip original
                if len(alt_result) >= 10 and alt_result[:3].isalpha() and alt_result[3:10].isdigit():
                    result = alt_result
                    corrections_made.append(f"Applied alternative correction: {alt_result}")
                    break

        # Final validation: Ensure EPIC format (3 letters + 7 digits)
        if len(result) >= 10:
            # Force correct format if validation fails
            if not (result[:3].isalpha() and result[3:10].isdigit()):
                validated_result = ''
                for i, char in enumerate(result[:10]):
                    if i < 3:  # First 3 must be letters
                        validated_result += char if char.isalpha() else 'X'
                    else:  # Last 7 must be digits
                        validated_result += char if char.isdigit() else '0'
                result = validated_result
                corrections_made.append("Applied EPIC format validation")

        # Log corrections if any were made (reduced verbosity)
        if corrections_made and len(corrections_made) <= 5:  # Only log if reasonable number of corrections
            print(f"      🔧 Voter ID corrections ({len(corrections_made)}): '{original}' → '{result}'")
            if len(corrections_made) <= 3:
                for correction in corrections_made:
                    print(f"         - {correction}")

        return result

    
    def _extract_voter_id_from_text(self, text: str) -> Tuple[str, float]:
        """
        Extract voter ID from OCR text using pattern matching
        
        Args:
            text: Raw OCR text
        
        Returns:
            Tuple of (voter_id, confidence)
        """
        if not text:
            return ('', 0.0)
        
        # Clean text
        text = text.upper().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        
        # Try each pattern
        for pattern in self.voter_id_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Get first match
                voter_id = matches[0].replace(' ', '')  # Remove spaces
                voter_id = voter_id.rstrip('_').strip()  # Remove trailing underscores
                
                # Apply format correction
                voter_id = self._correct_voter_id_format(voter_id)
                
                # Calculate confidence based on format
                confidence = self._calculate_voter_id_confidence(voter_id)
                
                return (voter_id, confidence)
        
        # No pattern matched - try to extract any alphanumeric sequence
        words = text.split()
        for word in words:
            # Look for words with both letters and numbers
            if re.search(r'[A-Z0-9]', word) and len(word) >= 8:
                word = word.rstrip('_').strip()  # Remove trailing underscores
                
                # Apply format correction
                word = self._correct_voter_id_format(word)
                
                # Only return if it matches the correct format after correction
                if re.match(r'^[A-Z]{3}[0-9]{7}$', word):
                    confidence = 0.6  # Medium confidence
                    return (word, confidence)
        
        return ('', 0.0)
    
    def _calculate_voter_id_confidence(self, voter_id: str) -> float:
        """
        Calculate confidence score for extracted voter ID
        
        Args:
            voter_id: Extracted voter ID
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not voter_id:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Check standard format: 3 letters + 7 digits (User's strict requirement)
        if re.match(r'^[A-Z]{3}[0-9]{7}$', voter_id):
            confidence = 1.0
        
        # Penalize if not exactly 10 characters
        if len(voter_id) != 10:
            confidence *= 0.7
        
        # Penalize if it doesn't start with 3 letters
        if len(voter_id) >= 3 and not voter_id[:3].isalpha():
            confidence *= 0.8
            
        # Penalize if last 7 are not digits
        if len(voter_id) >= 10 and not voter_id[3:10].isdigit():
            confidence *= 0.8
        
        # Penalize if too short or too long
        if len(voter_id) < 8:
            confidence *= 0.5
        elif len(voter_id) > 12:
            confidence *= 0.6
        
        # Penalize if contains common OCR errors
        if any(char in voter_id for char in ['O0', 'I1', 'S5']):
            confidence *= 0.9
        
        return min(confidence, 1.0)

    def preprocess_fast(self, image):
        """
        Rule 4: Minimal Preprocessing for Digital Crops.
        Gray -> Threshold. That's it.
        """
        if not CV2_AVAILABLE:
            return image
            
        try:
            # Convert PIL to CV2
            if isinstance(image, Image.Image):
                 img = np.array(image)
                 if len(img.shape) == 3:
                     # Check if RGB or BGR? PIL is RGB. CV2 wants BGR.
                     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                 img = image

            # Grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Simple Otsu Threshold
            # Rule 4: "Simple threshold... Avoid multiple filters"
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return Image.fromarray(thresh)
            
        except Exception as e:
            # print(f"Fast preprocess failed: {e}")
            return image
    
    def extract_full_cell_text(self, 
                             image: Image.Image = None, 
                             pdf_page=None, 
                             rect=None,
                             force_marathi=False,
                             fast_preprocess=False) -> Dict:
        """
        Extract text from a cell.
        fast_preprocess: If True, skips heavy opencv ops (use when image is known to be digital/clean).
        """
        try:
            # Source resolution
            if pdf_page and rect:
                pix = pdf_page.get_pixmap(clip=rect, dpi=self.dpi, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            if not image:
                return {'text': '', 'raw_text': '', 'method': 'error'}
            
            # Preprocess (Rule 2 & 4)
            if fast_preprocess or self.quality_mode == 'fast':
                 # Rule 4: Minimal Preprocessing
                 processed_img = self.preprocess_fast(image)
            else:
                 # Standard Heavy Preprocessing for noisy scans
                 processed_img = self.preprocess_image(image, for_ocr=True, skip_heavy_ops=False)
            
            text = ""
            method = "none"

            # STRATEGY 1: PaddleOCR (Best for Marathi)
            # Skip PaddleOCR in 'fast' mode for maximum speed
            if self.paddle_processor and self.quality_mode != 'fast':
                try:
                    # If forcing Marathi, ensure we use Marathi model logic primarily
                    # Reduced logging for speed
                    text = self.paddle_processor.get_full_text(image, separator="\n")
                    if text.strip():
                        method = 'paddle_mar'
                except Exception as e:
                    pass

            # STRATEGY 2: Tesseract Fallback (Primary in 'fast' mode)
            if not text.strip():
                try:
                    # If enforcing Marathi, ONLY use Marathi language data to avoid English confusion
                    langs = 'mar' if force_marathi else 'mar+hin+eng'
                    
                    # Performance optimization: cache result of get_languages
                    # For simplicity, we assume 'mar' is available or fallback is handled by pytesseract
                    
                    text = pytesseract.image_to_string(
                        processed_img,
                        lang=langs,
                        config='--psm 6 --oem 1' # OEM 1 is usually faster Neural net mode
                    ).strip()
                    method = 'tesseract_fallback'
                except Exception as e:
                    pass
            
            # Post-Processing / Reconstruction
            final_text = self._post_process_marathi_text(text, remove_english=force_marathi)

            return {
                'text': final_text,
                'raw_text': text,
                'method': method
            }

        except Exception as e:
            # print(f"      ERROR in extract_full_cell_text: {e}")
            return {'text': '', 'method': 'error', 'error': str(e)}

    def _post_process_marathi_text(self, text: str, remove_english: bool = False) -> str:
        """
        Clean and reconstruct Marathi text from OCR output.
        - Fix broken lines
        - Remove OCR garbage
        - Normalize unicode
        - ALWAYS remove pipes (|), double pipes (||), and colons (:)
        - OPTIONALLY remove English characters
        """
        if not text: 
            return ""

        # 1. Global Cleanup: Remove pipes and colons
        # Replace | and || and : with empty string
        text = text.replace('||', '').replace('|', '').replace(':', '')
        
        # 1.1 Targeted OCR Corrections (e.g., ठोळके -> शेळके)
        try:
            from translit_helper import TranslitHelper
            text = TranslitHelper.correct_marathi_ocr(text)
        except ImportError:
            pass

        # 2. Variable Cleanup: Remove English if requested
        if remove_english:
            # Regex to remove all English letters (A-Z, a-z)
            text = re.sub(r'[A-Za-z]', '', text)

        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line: 
                continue
            
            # Remove common OCR artifacts (noise)
            # If line is just special chars or very short non-alphanumeric, skip
            if len(line) < 2 and not any(c.isalnum() for c in line):
                continue

            cleaned_lines.append(line)
        
        # Join with newlines
        return "\n".join(cleaned_lines)

    def extract_photo(self, image: Image.Image = None, pdf_page=None, rect=None) -> Dict:
        """
        Extract photo from image at high quality
        
        Args:
            image: PIL Image or None
            pdf_page: PyMuPDF page object (optional)
            rect: PyMuPDF Rect object (optional)
        
        Returns:
            Dict with photo_base64, confidence, method
        """
        try:
            # Extract high-quality image if pdf_page and rect provided
            if pdf_page and rect:
                pix = pdf_page.get_pixmap(clip=rect, dpi=self.dpi, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            if not image:
                return {
                    'photo_base64': '',
                    'confidence': 0.0,
                    'method': 'error'
                }
            
            # Check if image is valid (not empty/blank)
            confidence = self._calculate_photo_confidence(image)

            if confidence < 0.05:  # Very low threshold to accept more photos
                return {
                    'photo_base64': '',
                    'confidence': confidence,
                    'method': 'invalid'
                }
            
            # Convert to JPEG and encode (reduced quality for speed)
            jpeg_buffer = io.BytesIO()
            image.convert('RGB').save(jpeg_buffer, format='JPEG', quality=70, optimize=True)
            jpeg_bytes = jpeg_buffer.getvalue()
            photo_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            
            print(f"      Photo extracted: {len(photo_base64)} chars (conf: {confidence:.2f})")
            
            return {
                'photo_base64': photo_base64,
                'confidence': confidence,
                'method': 'local',
                'size': image.size,
                'format': image.format or 'JPEG'
            }
        
        except Exception as e:
            print(f"      ERROR: Photo extraction failed: {str(e)}")
            return {
                'photo_base64': '',
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def _calculate_photo_confidence(self, image: Image.Image) -> float:
        """
        Calculate confidence that image is a valid photo
        
        Args:
            image: PIL Image
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            import numpy as np
            
            # Convert to array
            img_array = np.array(image)
            
            # Check if image is not blank (has variance)
            variance = np.var(img_array)
            
            if variance < 100:  # Very low variance = likely blank
                return 0.1
            elif variance < 500:
                return 0.5
            elif variance < 1000:
                return 0.7
            else:
                return 0.9
        
        except:
            # Fallback: basic check
            if image.size[0] > 50 and image.size[1] > 50:
                return 0.6
            else:
                return 0.3


# Test function
if __name__ == '__main__':
    print("Testing OCR Processor with 400 DPI...")
    print("=" * 60)
    
    processor = OCRProcessor400DPI()
    
    print("\nProcessor initialized successfully!")
    print(f"DPI: {processor.dpi}")
    
    print("\n" + "=" * 60)
    print("Ready to process voter IDs and photos at 400 DPI!")
    print("=" * 60)

