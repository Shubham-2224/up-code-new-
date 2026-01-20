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
        self.dpi = 300  # Modified to 300 DPI as requested
        self.current_lang = 'mr' # Default to Marathi
        
        # Configure Tesseract path from environment (critical for systemd services)
        tesseract_cmd = os.getenv('TESSERACT_CMD')
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif os.name != 'nt':  # Linux/Unix - ensure tesseract is findable
            for path in ['/usr/bin/tesseract', '/usr/local/bin/tesseract']:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        # Voter ID patterns (Indian EPIC format)
        self.voter_id_patterns = [
            r'\b[A-Z]{3}[0-9]{7}\b',           # Standard: ABC1234567
            r'\b[A-Z]{3}\s*[0-9]{7}\b',        # With space: ABC 1234567
            r'\b[A-Z]{2,4}[0-9]{6,8}\b',       # Flexible: 2-4 letters + 6-8 digits
        ]
        
        
        print("OK: OCR Processor initialized with 300 DPI")

        # Initialize PaddleOCR if available
        self.paddle_processor = None
        if PADDLE_PROCESSOR_AVAILABLE:
            try:
                self.paddle_processor = PaddleOCRProcessor(lang='mr')
            except Exception as e:
                print(f"Failed to init PaddleOCR: {e}")

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
                
                # 4. Sharpen
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(2.0)
                
                # 5. Contrast / Binarization
                # Initial High Contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)
                
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
            
            # Preprocess image
            processed_img = self.preprocess_image(image, for_ocr=True)
            
            # Use local Tesseract OCR with high quality settings
            # print(f"      OCR: Tesseract (300 DPI)...")
            raw_text = pytesseract.image_to_string(
                processed_img,
                lang='eng+hin',
                config='--psm 6 --oem 1 -c tessedit_do_invert=0'  # FAST mode, no invert
            ).strip()
            
            # print(f"      Raw OCR: '{raw_text[:50]}...'")
            
            # Extract voter ID using patterns
            voter_id, confidence = self._extract_voter_id_from_text(raw_text)
            
            # print(f"      Extracted: '{voter_id}' (confidence: {confidence:.2f})")
            
            # Return result
            # print(f"      ✓ Local OCR SUCCESS (conf: {confidence:.2f})")
            return {
                'voter_id': voter_id,
                'confidence': confidence,
                'method': 'tesseract',
                'raw_text': raw_text
            }
        
        except Exception as e:
            print(f"      ERROR: Voter ID extraction failed: {str(e)}")
            return {
                'voter_id': '',
                'confidence': 0.0,
                'method': 'error',
                'raw_text': '',
                'error': str(e)
            }
    
    def _correct_voter_id_format(self, voter_id: str) -> str:
        """
        Correct Voter ID format using strict rules:
        - Positions 1-3: Must be LETTERS (A-Z)
        - Positions 4-10: Must be NUMBERS (0-9)
        
        Args:
            voter_id: Raw voter ID from OCR
        
        Returns:
            Corrected voter ID
        """
        if not voter_id or len(voter_id) < 10:
            return voter_id
        
        original = voter_id
        
        # Letter correction map (for positions 1-3)
        # Numbers that look like letters -> Letters
        letter_corrections = {
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '3': 'E',
            '4': 'A',
            '5': 'S',
            '6': 'G',
            '7': 'T',
            '8': 'B',
            '9': 'G',
        }
        
        # Number correction map (for positions 4-10)
        # Letters that look like numbers -> Numbers
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
        
        corrected = list(voter_id)
        corrections_made = []
        
        # Phase 1: Correct first 3 positions (must be LETTERS)
        for i in range(min(3, len(corrected))):
            char = corrected[i]
            # If it's a digit, convert to letter
            if char.isdigit():
                new_char = letter_corrections.get(char, char)
                if new_char != char:
                    corrections_made.append(f"Pos {i+1}: '{char}' → '{new_char}' (letter)")
                    corrected[i] = new_char
        
        # Phase 2: Correct positions 4-10 (must be NUMBERS)
        for i in range(3, min(10, len(corrected))):
            char = corrected[i]
            # If it's a letter, convert to number
            if char.isalpha():
                new_char = number_corrections.get(char, char)
                if new_char != char:
                    corrections_made.append(f"Pos {i+1}: '{char}' → '{new_char}' (number)")
                    corrected[i] = new_char
        
        result = ''.join(corrected)
        
        # Log corrections if any were made
        if corrections_made:
            print(f"      📝 Voter ID Corrected: '{original}' → '{result}'")
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
        
        # Check standard format: 3 letters + 7 digits
        if re.match(r'^[A-Z]{3}[0-9]{7}$', voter_id):
            confidence = 0.95
        
        # Check flexible format: 2-4 letters + 6-8 digits
        elif re.match(r'^[A-Z]{2,4}[0-9]{6,8}$', voter_id):
            confidence = 0.85
        
        # Check if it has both letters and numbers
        elif re.search(r'[A-Z]', voter_id) and re.search(r'[0-9]', voter_id):
            confidence = 0.6
        
        # Penalize if too short or too long
        if len(voter_id) < 8:
            confidence *= 0.7
        elif len(voter_id) > 15:
            confidence *= 0.8
        
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
                # Rule 1: User suggested 250-300 DPI. We stick to 300 or reduce to 250 if needed.
                # Current class default is 300.
                pix = pdf_page.get_pixmap(clip=rect, dpi=self.dpi, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            if not image:
                return {'text': '', 'raw_text': '', 'method': 'error'}
            
            # Preprocess (Rule 2 & 4)
            if fast_preprocess:
                 # Rule 4: Minimal Preprocessing
                 processed_img = self.preprocess_fast(image)
            else:
                 # Standard Heavy Preprocessing for noisy scans
                 processed_img = self.preprocess_image(image, for_ocr=True, skip_heavy_ops=False)
            
            text = ""
            method = "none"

            # STRATEGY 1: PaddleOCR (Best for Marathi)
            if self.paddle_processor:
                try:
                    # If forcing Marathi, ensure we use Marathi model logic primarily
                    print(f"      OCR: Using PaddleOCR (Marathi) [Force Marathi: {force_marathi}]...")
                    text = self.paddle_processor.get_full_text(image, separator="\n")
                    if text.strip():
                        method = 'paddle_mar'
                except Exception as e:
                    print(f"      PaddleOCR failed: {e}")

            # STRATEGY 2: Tesseract Fallback
            if not text.strip():
                print("      OCR: Fallback to Tesseract...")
                try:
                    # If enforcing Marathi, ONLY use Marathi language data to avoid English confusion
                    langs = 'mar' if force_marathi else 'mar+hin+eng'
                    
                    found_langs = pytesseract.get_languages(config='')
                    if 'mar' not in found_langs:
                        print("      WARNING: Marathi language data not found, falling back to eng+hin")
                        langs = 'eng+hin'
                    
                    text = pytesseract.image_to_string(
                        processed_img,
                        lang=langs,
                        config='--psm 6' 
                    ).strip()
                    method = 'tesseract_fallback'
                except Exception as e:
                    print(f"      Tesseract failed: {e}")
            
            # Post-Processing / Reconstruction
            final_text = self._post_process_marathi_text(text, remove_english=force_marathi)

            return {
                'text': final_text,
                'raw_text': text,
                'method': method
            }

        except Exception as e:
            print(f"      ERROR in extract_full_cell_text: {e}")
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
            
            if confidence < 0.3:
                print(f"      WARNING: Photo appears blank or invalid (conf: {confidence:.2f})")
                return {
                    'photo_base64': '',
                    'confidence': confidence,
                    'method': 'invalid'
                }
            
            # Convert to JPEG and encode
            jpeg_buffer = io.BytesIO()
            image.convert('RGB').save(jpeg_buffer, format='JPEG', quality=90)
            jpeg_bytes = jpeg_buffer.getvalue()
            photo_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            
            print(f"      ✓ Photo extracted: {len(photo_base64)} chars (conf: {confidence:.2f})")
            
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

