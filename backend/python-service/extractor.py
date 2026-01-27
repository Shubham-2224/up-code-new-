"""
Enhanced Extractor - Grid-based extraction with local OCR
Uses local Tesseract OCR for high-accuracy extraction
"""

import os

# PERFORMANCE OPTIMIZATION: Suppress verbose logging BEFORE imports
os.environ['OMP_THREAD_LIMIT'] = '1'  # Limit Tesseract threads for better multiprocessing
os.environ['GLOG_minloglevel'] = '3'  # Suppress PaddlePaddle C++ logs (0=INFO, 3=ERROR)
os.environ['FLAGS_print_model_net_proto'] = '0'  # Don't print model proto
os.environ['PADDLEOCR_SHOW_LOG'] = '0'  # Suppress PaddleOCR logs

# Memory optimization settings
os.environ['PYTHONUNBUFFERED'] = '0'  # Reduce I/O buffering
import gc  # Enable garbage collection control

import fitz  # PyMuPDF
import pytesseract
import base64
import io
from PIL import Image
import re
from typing import Dict, List, Optional
import multiprocessing as mp
from functools import partial
import time

# Import advanced modules
try:
    from photo_processor import PhotoProcessor
    PHOTO_PROCESSOR_AVAILABLE = True
except ImportError:
    PHOTO_PROCESSOR_AVAILABLE = False
    pass  # Photo Processor optional

try:
    from box_detector import BoxDetector
    BOX_DETECTOR_AVAILABLE = True
except ImportError:
    BOX_DETECTOR_AVAILABLE = False
    pass  # Box Detector optional

try:
    from smart_detector import SmartDetector
    SMART_DETECTOR_AVAILABLE = True
except ImportError:
    SMART_DETECTOR_AVAILABLE = False
    pass  # Smart Detector optional
    
from translit_helper import TranslitHelper

# Import 400 DPI OCR Processor
try:
    from ocr_processor_400dpi import OCRProcessor400DPI
    OCR_400DPI_AVAILABLE = True
except ImportError:
    OCR_400DPI_AVAILABLE = False
    pass  # 400 DPI OCR Processor optional

# Configure Tesseract OCR path
# Priority: 1. TESSERACT_CMD env var, 2. Auto-detect by OS
tesseract_cmd_from_env = os.getenv('TESSERACT_CMD')
if tesseract_cmd_from_env:
    # Use explicitly set path from environment (important for systemd services)
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_from_env
elif os.name == 'nt':  # Windows - auto-detect
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
else:
    # Linux/Unix - try common locations if not in PATH
    linux_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/bin/tesseract'
    ]
    for path in linux_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

# Initialize processors
photo_processor = PhotoProcessor() if PHOTO_PROCESSOR_AVAILABLE else None
box_detector = BoxDetector() if BOX_DETECTOR_AVAILABLE else None
smart_detector = SmartDetector() if SMART_DETECTOR_AVAILABLE else None
ocr_processor_400dpi = OCRProcessor400DPI() if OCR_400DPI_AVAILABLE else None

# Pre-compile regex patterns for better performance
EPIC_REGEX = re.compile(r'^[A-Z]{3}[0-9]{7}$')
LOOSE_EPIC_REGEX = re.compile(r'^[A-Z]{2,4}[0-9]{6,8}$')
DEVANAGARI_REGEX = re.compile(r'[\u0900-\u097F]')
NUMBER_CLEANUP_REGEX = re.compile(r'[0-9,]+')

# Get CPU count for multiprocessing
def get_cpu_count():
    """Get optimal CPU count for multiprocessing - USE ALL CORES!"""
    try:
        cpu_count = mp.cpu_count()
        # Use 100% of available CPUs for maximum speed
        optimal_workers = max(1, cpu_count)
        return optimal_workers
    except:
        return 4  # Fallback to 4 workers

CPU_WORKERS = get_cpu_count()
# CPU optimization enabled - using all cores

def _extract_cell_internal(page, page_num, cell_info, config, extraction_limits, processors, master_page_img=None, master_page_scale=None):
    """
    Internal function to process a cell given an open page and processors.
    Refactored for page-level optimization.
    """
    try:
        extraction_y_start, extraction_y_end = extraction_limits
        
        # Extract cell info
        cell_x = cell_info['x']
        cell_y = cell_info['y']
        cell_width_actual = cell_info['width']
        cell_height_actual = cell_info['height']
        row = cell_info['row']
        col = cell_info['col']
        scale_x = cell_info['scale_x']
        scale_y = cell_info['scale_y']
        first_cell_width = cell_info.get('first_cell_width', cell_width_actual)
        first_cell_height = cell_info.get('first_cell_height', cell_height_actual)
        
        # Skip if cell is COMPLETELY outside the extraction area (header/footer zone)
        cell_bottom = cell_y + cell_height_actual
        
        # Skip only if cell is completely in header zone OR completely in footer zone
        if cell_bottom <= extraction_y_start or cell_y >= extraction_y_end:
            return None
        
        # Get configuration
        cell_template = config.get('cellTemplate', {})
        voter_id_box = cell_template.get('voterIdBox', {})
        photo_box = cell_template.get('photoBox', {})

        # Cache performance settings
        extract_photos = config.get('extractPhotos', True)
        performance_mode = config.get('performanceMode', 'balanced')  # fast, balanced, accurate

        # Get processors (only get photo processor if needed)
        local_ocr_processor = processors.get('ocr')
        local_photo_processor = processors.get('photo') if extract_photos else None
        local_smart_detector = processors.get('smart')
        
        # === EXTRACT VOTER ID WITH MULTI-STRATEGY APPROACH ===
        voter_id_text = ""
        voter_id_confidence = 0.0
        voter_id_method = "none"
        cell_stats = {}
        
        # STRATEGY 0: Try PDF Text Layer First (FASTEST & MOST ACCURATE!)
        if voter_id_box:
            try:
                scaled_voter_id_x = voter_id_box.get('x', 0) * scale_x
                scaled_voter_id_y = voter_id_box.get('y', 0) * scale_y
                scaled_voter_id_width = voter_id_box.get('width', 200) * scale_x
                scaled_voter_id_height = voter_id_box.get('height', 30) * scale_y
                
                voter_id_rect = fitz.Rect(
                    cell_x + scaled_voter_id_x,
                    cell_y + scaled_voter_id_y,
                    cell_x + scaled_voter_id_x + scaled_voter_id_width,
                    cell_y + scaled_voter_id_y + scaled_voter_id_height
                )
                
                # Extract text from PDF text layer
                if VERBOSE_OCR_LOGS:
                    print(f"      📄 Strategy 0: Extracting from PDF text layer...")
                text_layer = page.get_text("text", clip=voter_id_rect).strip()
                
                if text_layer:
                    if VERBOSE_OCR_LOGS:
                        print(f"      📝 Text layer found: '{text_layer[:50]}...'")
                    
                    # Clean and extract voter ID from text layer
                    text_layer_clean = text_layer.upper().strip()
                    text_layer_clean = re.sub(r'[^A-Z0-9\s]', '', text_layer_clean)  # Remove special chars
                    text_layer_clean = ' '.join(text_layer_clean.split())  # Normalize whitespace
                    
                    # Try to find voter ID pattern in text layer
                    voter_id_patterns = [
                        r'\b([A-Z]{3}[0-9]{7})\b',           # Standard: ABC1234567
                        r'\b([A-Z]{3}\s*[0-9]{7})\b',        # With space: ABC 1234567
                        r'\b([A-Z]{2,4}[0-9]{6,8})\b',       # Flexible
                    ]
                    
                    text_layer_voter_id = ""
                    for pattern in voter_id_patterns:
                        matches = re.findall(pattern, text_layer_clean)
                        if matches:
                            text_layer_voter_id = matches[0].replace(' ', '')
                            break
                    
                    # If found valid voter ID in text layer, use it!
                    if text_layer_voter_id and len(text_layer_voter_id) >= 10:
                        # Apply format correction
                        if local_ocr_processor:
                            text_layer_voter_id = local_ocr_processor._correct_voter_id_format(text_layer_voter_id)
                        
                        voter_id_text = text_layer_voter_id
                        voter_id_confidence = 1.0  # Text layer is 100% accurate
                        voter_id_method = 'pdf_text_layer'
                        cell_stats['pdf_text_layer'] = 1
                        if VERBOSE_OCR_LOGS:
                            print(f"      ✅ PDF Text Layer SUCCESS: '{voter_id_text}' (conf=1.00)")
                    elif VERBOSE_OCR_LOGS:
                        print(f"      ⚠️  Text layer found but no valid voter ID pattern: '{text_layer_clean}'")
                else:
                    if VERBOSE_OCR_LOGS:
                        print(f"      ℹ️  No text layer found, will try OCR...")
                    
            except Exception as e:
                if VERBOSE_OCR_LOGS:
                    print(f"      ⚠️  Text layer extraction error: {str(e)}")
        
        # Strategy 1: Use 400 DPI OCR Processor (if text layer failed)
        if (not voter_id_text or len(voter_id_text.strip()) < 3) and local_ocr_processor and voter_id_box:
            try:
                scaled_voter_id_x = voter_id_box.get('x', 0) * scale_x
                scaled_voter_id_y = voter_id_box.get('y', 0) * scale_y
                scaled_voter_id_width = voter_id_box.get('width', 200) * scale_x
                scaled_voter_id_height = voter_id_box.get('height', 30) * scale_y
                
                voter_id_rect = fitz.Rect(
                    cell_x + scaled_voter_id_x,
                    cell_y + scaled_voter_id_y,
                    cell_x + scaled_voter_id_x + scaled_voter_id_width,
                    cell_y + scaled_voter_id_y + scaled_voter_id_height
                )
                
                if VERBOSE_OCR_LOGS:
                    print(f"      📄 Strategy 1: OCR {local_ocr_processor.dpi} DPI ({performance_mode} mode)...")

                # OPTIMIZATION: Crop directly from master_page_img if available
                voter_id_crop = None
                if master_page_img:
                    left = voter_id_rect.x0 * master_page_scale
                    top = voter_id_rect.y0 * master_page_scale
                    right = voter_id_rect.x1 * master_page_scale
                    bottom = voter_id_rect.y1 * master_page_scale
                    voter_id_crop = master_page_img.crop((left, top, right, bottom))

                result = local_ocr_processor.extract_voter_id(
                    image=voter_id_crop,
                    pdf_page=None if voter_id_crop else page,
                    rect=None if voter_id_crop else voter_id_rect
                )

                voter_id_text = result.get('voter_id', '')
                voter_id_confidence = result.get('confidence', 0.0)
                voter_id_method = result.get('method', 'unknown')

                # Early success check based on performance mode
                min_confidence = local_ocr_processor.min_confidence_threshold
                if voter_id_confidence >= min_confidence and voter_id_text:
                    if VERBOSE_OCR_LOGS:
                        print(f"      ✅ {performance_mode.upper()} MODE: Early success with conf={voter_id_confidence:.2f}")
                    cell_stats['ocr_400dpi'] = 1
                    # Skip to final processing
                    voter_id_text = voter_id_text if voter_id_text else ""
                
                if voter_id_method == 'tesseract':
                    cell_stats['ocr_400dpi_local'] = 1
                
                # RETRY LOGIC: If first attempt failed or low confidence, try again with higher DPI
                # SKIP RETRY IN FAST MODE to save significant time
                if performance_mode != 'fast' and (not voter_id_text or len(voter_id_text.strip()) < 3 or voter_id_confidence < 0.7):
                    if VERBOSE_OCR_LOGS:
                        print(f"      🔄 OCR Retry: First attempt failed/low confidence (conf={voter_id_confidence:.2f})")
                        print(f"      📄 Strategy 2: 450 DPI with enhanced preprocessing...")
                    
                    try:
                        # Extract with HIGHER DPI (450)
                        voter_id_pix = page.get_pixmap(clip=voter_id_rect, dpi=450, alpha=False)
                        
                        # FAST CONVERSION: Avoid PNG encode/decode overhead
                        if voter_id_pix.n < 4:
                            voter_id_img = Image.frombytes("RGB", [voter_id_pix.width, voter_id_pix.height], voter_id_pix.samples)
                        else:
                             voter_id_img = Image.frombytes("RGB", [voter_id_pix.width, voter_id_pix.height], voter_id_pix.samples)

                        # Enhanced preprocessing for retry
                        from PIL import ImageEnhance, ImageFilter
                        
                        # Convert to grayscale
                        voter_id_img = voter_id_img.convert('L')
                        
                        # Increase contrast more aggressively
                        enhancer = ImageEnhance.Contrast(voter_id_img)
                        voter_id_img = enhancer.enhance(2.5)
                        
                        # Sharpen
                        enhancer = ImageEnhance.Sharpness(voter_id_img)
                        voter_id_img = enhancer.enhance(2.0)
                        
                        # Denoise
                        voter_id_img = voter_id_img.filter(ImageFilter.MedianFilter(size=3))
                        
                        # Try multiple PSM modes with EARLY EXIT
                        retry_texts = []
                        best_retry_confidence = 0.0
                        
                        for psm_mode in [6, 7, 8, 11, 13]:  # Try MORE page segmentation modes
                            try:
                                retry_raw_text = pytesseract.image_to_string(
                                    voter_id_img,
                                    lang='eng+hin',
                                    config=f'--psm {psm_mode} --oem 3'
                                ).strip()
                                
                                if retry_raw_text:
                                    retry_texts.append(retry_raw_text)
                                    
                                    # EARLY EXIT: If we found a good result, stop trying other modes!
                                    curr_result = local_ocr_processor._extract_voter_id_from_text(retry_raw_text)
                                    if curr_result[0] and len(curr_result[0].strip()) >= 10 and curr_result[1] > 0.85:
                                        if VERBOSE_OCR_LOGS:
                                            print(f"      ⚡ Fast Match (PSM {psm_mode}): '{curr_result[0]}' (conf={curr_result[1]:.2f})")
                                        break
                            except:
                                pass
                        
                        # Use the result
                        if retry_texts:
                            # Re-extract and find best
                            final_retry_voter_id = ""
                            final_retry_confidence = 0.0
                            
                            for text in retry_texts:
                                res = local_ocr_processor._extract_voter_id_from_text(text)
                                if res[1] > final_retry_confidence:
                                    final_retry_confidence = res[1]
                                    final_retry_voter_id = res[0]

                            # Use retry result if it's better
                            if final_retry_voter_id and len(final_retry_voter_id.strip()) >= 3:
                                if final_retry_confidence > voter_id_confidence or not voter_id_text:
                                    if VERBOSE_OCR_LOGS:
                                        print(f"      ✅ Strategy 2 SUCCESS: '{final_retry_voter_id}' (conf={final_retry_confidence:.2f})")
                                    voter_id_text = final_retry_voter_id
                                    voter_id_confidence = final_retry_confidence
                                    voter_id_method = 'tesseract_retry_600dpi'
                                    cell_stats['ocr_retry_600dpi'] = 1
                                else:
                                    if VERBOSE_OCR_LOGS:
                                        print(f"      ⚠️  Strategy 2 result not better: '{final_retry_voter_id}' (conf={final_retry_confidence:.2f})")
                            else:
                                if VERBOSE_OCR_LOGS:
                                    print(f"      ❌ Strategy 2 failed: No valid voter ID found")
                    except Exception as retry_error:
                        if VERBOSE_OCR_LOGS:
                            print(f"      ❌ Strategy 2 error: {str(retry_error)}")
                
                # ULTRA-AGGRESSIVE FALLBACK: If still failed (or low confidence), try MAXIMUM quality
                # OPTIMIZATION: Only run if confidence is really low (< 0.75) to save time
                if (not voter_id_text or len(voter_id_text.strip()) < 3 or voter_id_confidence < 0.75):
                    if VERBOSE_OCR_LOGS:
                        print(f"      🔥 ULTRA-AGGRESSIVE FALLBACK: Trying MAXIMUM quality extraction...")
                        print(f"      📄 Strategy 3: 800 DPI + ALL PSM modes + Multiple preprocessing...")
                    
                    try:
                        # Extract with HIGH DPI (500) - Lowered from 800 for speed
                        # 500 DPI with Binarization is often better than 800 DPI raw
                        voter_id_pix_ultra = page.get_pixmap(clip=voter_id_rect, dpi=500, alpha=False)
                        # FAST BUFFER CONVERSION
                        voter_id_img_ultra = Image.frombytes("RGB", [voter_id_pix_ultra.width, voter_id_pix_ultra.height], voter_id_pix_ultra.samples)
                        
                        from PIL import ImageEnhance, ImageFilter, ImageOps
                        
                        # Try MULTIPLE preprocessing variants
                        preprocessing_variants = []
                        
                        # Variant 1: High contrast + sharp
                        img1 = voter_id_img_ultra.convert('L')
                        img1 = ImageEnhance.Contrast(img1).enhance(3.0)
                        img1 = ImageEnhance.Sharpness(img1).enhance(2.5)
                        preprocessing_variants.append(('high_contrast', img1))
                        
                        # Variant 2: Inverted (white text on black)
                        img2 = voter_id_img_ultra.convert('L')
                        img2 = ImageOps.invert(img2)
                        img2 = ImageEnhance.Contrast(img2).enhance(2.0)
                        preprocessing_variants.append(('inverted', img2))
                        
                        # Variant 3: Threshold (binary)
                        img3 = voter_id_img_ultra.convert('L')
                        img3 = img3.point(lambda x: 0 if x < 128 else 255, '1')
                        preprocessing_variants.append(('threshold', img3))
                        
                        # Variant 4: Adaptive threshold
                        img4 = voter_id_img_ultra.convert('L')
                        img4 = ImageEnhance.Contrast(img4).enhance(2.0)
                        img4 = img4.filter(ImageFilter.SHARPEN)
                        preprocessing_variants.append(('adaptive', img4))
                        
                        # Try ALL PSM modes on ALL preprocessing variants WITH EARLY EXIT!
                        best_ultra_result = None
                        best_ultra_confidence = 0.0
                        found_good_match = False
                        
                        psm_modes = [3, 4, 6, 7, 8, 11, 12, 13]  # ALL useful PSM modes
                        
                        for variant_name, variant_img in preprocessing_variants:
                            if found_good_match: break
                            
                            for psm_mode in psm_modes:
                                try:
                                    ultra_raw_text = pytesseract.image_to_string(
                                        variant_img,
                                        lang='eng+hin',
                                        config=f'--psm {psm_mode} --oem 3'
                                    ).strip()
                                    
                                    if ultra_raw_text:
                                        # Check right away!
                                        ultra_result = local_ocr_processor._extract_voter_id_from_text(ultra_raw_text)
                                        ultra_voter_id = ultra_result[0]
                                        ultra_confidence = ultra_result[1]
                                        
                                        if ultra_voter_id and len(ultra_voter_id.strip()) >= 10:
                                            # Keep track of best
                                            if ultra_confidence > best_ultra_confidence:
                                                best_ultra_result = (variant_name, psm_mode, ultra_voter_id, ultra_confidence)
                                                best_ultra_confidence = ultra_confidence
                                            
                                            # EARLY EXIT: If confidence is high, STOP searching!
                                            if ultra_confidence > 0.90:
                                                if VERBOSE_OCR_LOGS:
                                                    print(f"      ⚡ Fast Match (Ultra): {variant_name} + PSM {psm_mode} -> (conf={ultra_confidence:.2f})")
                                                found_good_match = True
                                                break
                                except:
                                    pass
                            
                        if best_ultra_result:
                            variant_name, psm_mode, ultra_voter_id, ultra_confidence = best_ultra_result
                            if VERBOSE_OCR_LOGS:
                                print(f"      ✅ ULTRA-FALLBACK SUCCESS: '{ultra_voter_id}' (conf={ultra_confidence:.2f})")
                                print(f"         Best: {variant_name} preprocessing + PSM {psm_mode}")
                            voter_id_text = ultra_voter_id
                            voter_id_confidence = ultra_confidence
                            voter_id_method = 'tesseract_ultra_800dpi'
                            cell_stats['ocr_ultra_800dpi'] = 1
                        else:
                            if VERBOSE_OCR_LOGS:
                                print(f"      ❌ ULTRA-FALLBACK: No valid voter ID found in up to 32 attempts")
                    except Exception as ultra_error:
                        if VERBOSE_OCR_LOGS:
                            print(f"      ❌ ULTRA-FALLBACK error: {str(ultra_error)}")
                
            except Exception as e:
                if VERBOSE_OCR_LOGS:
                    print(f"      ❌ OCR error: {str(e)}")
                voter_id_text = ""
                voter_id_confidence = 0.0
        
        # Strategy 2: Fallback to legacy method (if 400 DPI not available)
        elif voter_id_box:
            try:
                scaled_voter_id_x = voter_id_box.get('x', 0) * scale_x
                scaled_voter_id_y = voter_id_box.get('y', 0) * scale_y
                scaled_voter_id_width = voter_id_box.get('width', 200) * scale_x
                scaled_voter_id_height = voter_id_box.get('height', 30) * scale_y
                
                voter_id_rect = fitz.Rect(
                    cell_x + scaled_voter_id_x,
                    cell_y + scaled_voter_id_y,
                    cell_x + scaled_voter_id_x + scaled_voter_id_width,
                    cell_y + scaled_voter_id_y + scaled_voter_id_height
                )
                
                voter_id_pix = page.get_pixmap(clip=voter_id_rect, dpi=300, alpha=False)
                # FAST BUFFER CONVERSION
                voter_id_img = Image.frombytes("RGB", [voter_id_pix.width, voter_id_pix.height], voter_id_pix.samples)
                
                raw_text = pytesseract.image_to_string(
                    voter_id_img,
                    lang='eng+hin',
                    config='--psm 6'
                ).strip()
                
                voter_id_text = clean_voter_id(raw_text)
                voter_id_confidence = 0.5
                cell_stats['tesseract_ocr'] = 1
            except Exception as e:
                voter_id_text = ""
                voter_id_confidence = 0.0
        
        # Strategy 3: Smart Detection
        elif local_smart_detector:
            try:
                cell_rect = fitz.Rect(cell_x, cell_y, cell_x + cell_width_actual, cell_y + cell_height_actual)
                cell_pix = page.get_pixmap(clip=cell_rect, dpi=200, alpha=False)
                # FAST BUFFER CONVERSION
                cell_img = Image.frombytes("RGB", [cell_pix.width, cell_pix.height], cell_pix.samples)
                
                smart_result = local_smart_detector.find_voter_id_in_cell(cell_img)
                if smart_result['found']:
                    voter_id_text = smart_result['voter_id']
                    voter_id_confidence = smart_result['confidence']
                    cell_stats['smart_voter_id_found'] = 1
            except:
                pass
        
        # === EXTRACT PHOTO (Only if enabled) ===
        photo_base64 = ""
        photo_quality = 0.0
        photo_method = "none"

        if extract_photos:
            # Strategy 1: Use 400 DPI OCR Processor (only if photo extraction enabled)
            if local_ocr_processor and photo_box:
                try:
                    scaled_photo_x = photo_box.get('x', 0) * scale_x
                    scaled_photo_y = photo_box.get('y', 0) * scale_y
                    scaled_photo_width = photo_box.get('width', 150) * scale_x
                    scaled_photo_height = photo_box.get('height', 180) * scale_y
                    
                    photo_rect = fitz.Rect(
                        cell_x + scaled_photo_x,
                        cell_y + scaled_photo_y,
                        cell_x + scaled_photo_x + scaled_photo_width,
                        cell_y + scaled_photo_y + scaled_photo_height
                    )
                    
                    # OPTIMIZATION: Crop from master image for perfect alignment
                    photo_crop = None
                    if master_page_img:
                        left = photo_rect.x0 * master_page_scale
                        top = photo_rect.y0 * master_page_scale
                        right = photo_rect.x1 * master_page_scale
                        bottom = photo_rect.y1 * master_page_scale
                        photo_crop = master_page_img.crop((left, top, right, bottom))

                    result = local_ocr_processor.extract_photo(
                        image=photo_crop,
                        pdf_page=None if photo_crop else page,
                        rect=None if photo_crop else photo_rect
                    )
                    
                    photo_base64 = result.get('photo_base64', '')
                    photo_quality = result.get('confidence', 0.0)
                    photo_method = result.get('method', 'unknown')
                    
                    # Enhance photo if processor available
                    if photo_base64 and local_photo_processor:
                        try:
                            img_bytes = base64.b64decode(photo_base64)
                            img = Image.open(io.BytesIO(img_bytes))
                            photo_result = local_photo_processor.process_photo(img, enhance=True, resize=False)
                            photo_base64 = photo_result['base64']
                            photo_quality = photo_result.get('quality_score', photo_quality)
                            cell_stats['photo_enhanced'] = 1
                        except:
                            pass
                    
                    if photo_base64:
                        cell_stats['photo_400dpi'] = 1
                except:
                    photo_base64 = ""
                    photo_quality = 0.0
            
            # Strategy 2: Fallback to legacy method
            elif photo_box:
                try:
                    scaled_photo_x = photo_box.get('x', 0) * scale_x
                    scaled_photo_y = photo_box.get('y', 0) * scale_y
                    scaled_photo_width = photo_box.get('width', 150) * scale_x
                    scaled_photo_height = photo_box.get('height', 180) * scale_y
                    
                    photo_rect = fitz.Rect(
                        cell_x + scaled_photo_x,
                        cell_y + scaled_photo_y,
                        cell_x + scaled_photo_x + scaled_photo_width,
                        cell_y + scaled_photo_y + scaled_photo_height
                    )
                    
                    photo_pix = page.get_pixmap(clip=photo_rect, dpi=300, alpha=False)
                    # FAST BUFFER CONVERSION
                    photo_img = Image.frombytes("RGB", [photo_pix.width, photo_pix.height], photo_pix.samples)
                    
                    if local_photo_processor:
                        photo_result = local_photo_processor.process_photo(photo_img, enhance=True, resize=False)
                        photo_base64 = photo_result['base64']
                        photo_quality = photo_result.get('quality_score', 0.5)
                        cell_stats['photo_enhanced'] = 1
                    else:
                        jpeg_buffer = io.BytesIO()
                        photo_img.convert('RGB').save(jpeg_buffer, format='JPEG', quality=85)
                        jpeg_bytes = jpeg_buffer.getvalue()
                        photo_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
                        photo_quality = 0.5
                except:
                    photo_base64 = ""
                    photo_quality = 0.0
            
            # Strategy 3: Smart Detection
            elif local_smart_detector:
                try:
                    cell_rect = fitz.Rect(cell_x, cell_y, cell_x + cell_width_actual, cell_y + cell_height_actual)
                    cell_pix = page.get_pixmap(clip=cell_rect, dpi=200, alpha=False)
                    # FAST BUFFER CONVERSION
                    cell_img = Image.frombytes("RGB", [cell_pix.width, cell_pix.height], cell_pix.samples)
                    
                    smart_result = local_smart_detector.find_photo_in_cell(cell_img)
                    if smart_result['found']:
                        photo_base64 = smart_result['photo_base64']
                        photo_quality = smart_result['confidence']
                        cell_stats['smart_photo_found'] = 1
                except:
                    pass
        
        # === FINAL VERIFICATION: HIGH-POWER TEXT LAYER MATCHING (99% ACCURACY) ===
        # User requested maximum accuracy using CPU power to match PDF text layer with extracted data.
        # We now use 'words' extraction to get precise coordinates of every text element.
        try:
            # Define cell verify rect
            cell_rect_verify = fitz.Rect(
                cell_x, 
                cell_y, 
                cell_x + cell_width_actual, 
                cell_y + cell_height_actual
            )
            
            # Extract ALL words with coordinates: (x0, y0, x1, y1, "word", block, line, word)
            # This uses more CPU but gives perfect spatial awareness
            cell_words = page.get_text("words", clip=cell_rect_verify)
            
            # Compile candidates with their metadata
            candidates = []
            
            # Expected Voter ID location (relative to cell)
            expected_x = cell_x + (voter_id_box.get('x', 0) * scale_x)
            expected_y = cell_y + (voter_id_box.get('y', 0) * scale_y)
            
            # Standard Pattern for Voter ID
            voter_id_pattern = EPIC_REGEX
            
            for w in cell_words:
                w_text = w[4].strip().upper()
                w_x0, w_y0 = w[0], w[1]
                
                # Check for Voter ID patterns (Standard)
                if voter_id_pattern.match(w_text):
                    dist = ((w_x0 - expected_x)**2 + (w_y0 - expected_y)**2)**0.5
                    candidates.append({'text': w_text, 'conf': 1.0, 'dist': dist, 'type': 'exact'})
            
            # Advanced Pattern Search: Concatenate adjacent words to handle split IDs
            # Example: "UZZ" + "1234567" or "UZZ123" + "4567"
            for i in range(len(cell_words)-1):
                w1 = cell_words[i]
                w2 = cell_words[i+1]
                
                # If words are physically close (same line approximation)
                if abs(w1[1] - w2[1]) < 5 and abs(w2[0] - w1[2]) < 10:
                    combined = (w1[4] + w2[4]).replace(' ', '').upper()
                    if voter_id_pattern.match(combined):
                        w_x0, w_y0 = w1[0], w1[1]
                        dist = ((w_x0 - expected_x)**2 + (w_y0 - expected_y)**2)**0.5
                        candidates.append({'text': combined, 'conf': 0.99, 'dist': dist, 'type': 'combined'})

            if candidates:
                # Sort by distance to expected location (Primary sorting)
                candidates.sort(key=lambda x: x['dist'])
                
                best_match = candidates[0]['text']
                match_type = candidates[0]['type']
                match_dist = candidates[0]['dist']
                
                if VERBOSE_OCR_LOGS:
                    print(f"      🛡️  Deep Verify Found: {best_match} (dist={match_dist:.1f}, type={match_type})")
                
                # LOGIC: Text Layer is the Ground Truth (99% Accuracy Source)
                current_nospaces = voter_id_text.replace(' ', '').strip().upper() if voter_id_text else ""
                
                # If we found a text layer match that looks like a Voter ID, use it.
                # STRICTER TOLERANCE: Reduced from 200 to 75 to ensure we don't accidentally pick a neighbor's ID
                # This ensures "5 is 5" and "6 is 6" by verifying we are looking at the EXACT right text element.
                if match_dist < 75:
                    if best_match != current_nospaces:
                        if current_nospaces:
                             print(f"      ⚠️  MISMATCH DETECTED. Starting Step-by-Step Digit Check:")
                             
                             # Compare Digit by Digit
                             max_len = max(len(best_match), len(current_nospaces))
                             txt_padded = best_match.ljust(max_len)
                             ocr_padded = current_nospaces.ljust(max_len)
                             
                             for i in range(max_len):
                                 c_txt = txt_padded[i]
                                 c_ocr = ocr_padded[i]
                                 
                                 if c_txt != c_ocr:
                                     print(f"         Step {i+1}: OCR '{c_ocr}' != Text '{c_txt}' -> ENFORCING Text Layer '{c_txt}'")
                                     if (c_ocr == '6' and c_txt == '5') or (c_ocr == '5' and c_txt == '6'):
                                         print(f"                  🛡️  CRITICAL FIX: 5/6 Ambiguity Resolved. Strictly using '{c_txt}'")
                             
                             print(f"      ✅ CORRECTION COMPLETE: Replaced '{current_nospaces}' with '{best_match}'")
                        else:
                             print(f"      ✅ RECOVERY: Found '{best_match}' from TextLayer")
                        
                        voter_id_text = best_match
                        voter_id_confidence = 1.0
                        voter_id_method = f'text_layer_deep_{match_type}'
                        cell_stats[f'text_layer_deep_{match_type}'] = 1
                    else:
                        print(f"      ✨ PERFECT INTEGRITY: '{best_match}' is identical in OCR and Text Layer.")
                        voter_id_confidence = 1.0
                else:
                    print(f"      ⚠️  Ignored Text Layer match (too far: {match_dist:.1f}px) - Risk of mismatch")
            
            else:
                 pass # No text layer candidate found
                 
        except Exception as e:
            print(f"      ⚠️  Deep Verification Error: {str(e)}")

        # Clean voter ID
        if voter_id_text:
            # Normalize: Uppercase and remove punctuation/spaces
            voter_id_text = re.sub(r'[^A-Z0-9]', '', voter_id_text.upper())
            
            # STRICT REQUIREMENT: ABC1234567 (10 chars: 3 Alpha + 7 Numeric)
            if len(voter_id_text) > 10:
                voter_id_text = voter_id_text[:10]
            
            # Apply format-specific fixes if we have exactly 10 characters
            if len(voter_id_text) == 10:
                # Ensure positions 1-3 are Alpha
                prefix = voter_id_text[:3]
                suffix = voter_id_text[3:]
                
                # Fix common OCR digit-to-letter errors in prefix
                prefix = prefix.replace('0', 'O').replace('1', 'I').replace('2', 'Z').replace('5', 'S').replace('8', 'B')
                
                # Fix common OCR letter-to-digit errors in suffix
                suffix = suffix.replace('O', '0').replace('I', '1').replace('L', '1').replace('S', '5').replace('G', '6').replace('B', '8').replace('Z', '2')
                
                voter_id_text = prefix + suffix
        
        # DEBUG: Log raw voter ID for troubleshooting
        print(f"      🔍 DEBUG Page {page_num+1}, Row {row+1}, Col {col+1}:")
        print(f"         Raw Voter ID: '{voter_id_text}'")
        print(f"         Voter ID Length: {len(voter_id_text) if voter_id_text else 0}")
        print(f"         Voter ID Confidence: {voter_id_confidence:.2f}")
        print(f"         Has Photo: {bool(photo_base64 and len(photo_base64) > 0)}")
        
        # Skip logic - EXTRACT EVERYTHING WITH DATA!
        # RULE: Never skip if we have ANY data (voter ID OR photo)
        should_skip = False
        skip_reason = ""
        
        # Check what data we have - LESS STRICT VALIDATION
        # Accept ANY voter ID that has content and isn't explicitly marked as invalid
        invalid_voter_ids = ["NO ID", "NOID", "N/A", "NA", "NOT FOUND", "NONE", "NULL", ""]
        
        # Check if voter ID is valid (has content and not in invalid list)
        has_valid_voter_id = False
        if voter_id_text:
            voter_id_upper = voter_id_text.upper().strip()
            # Accept if it's not in the invalid list and has at least 3 characters
            if voter_id_upper not in invalid_voter_ids and len(voter_id_text.strip()) >= 3:
                has_valid_voter_id = True
                print(f"         ✓ Voter ID is VALID: '{voter_id_text}'")
            else:
                print(f"         ✗ Voter ID is INVALID: '{voter_id_text}' (reason: {'too short' if len(voter_id_text.strip()) < 3 else 'in invalid list'})")
        else:
            print(f"         ✗ Voter ID is EMPTY")
        
        has_photo = photo_base64 and len(photo_base64) > 0
        
        # CRITICAL: If we have a photo, NEVER skip (even without voter ID)
        if has_photo:
            should_skip = False  # Always extract if photo exists
            print(f"         → Decision: EXTRACT (has photo)")
        # If we have voter ID but no photo, still extract
        elif has_valid_voter_id:
            should_skip = False  # Always extract if voter ID exists
            print(f"         → Decision: EXTRACT (has valid voter ID)")
        # Only skip if we have NOTHING
        else:
            should_skip = True
            skip_reason = f"No valid voter ID (got: '{voter_id_text}')"
            print(f"         → Decision: SKIP ({skip_reason})")
        
        # USER REQUEST 2026-01-27: REMOVE STRICT FILTERING
        # User wants ALL data extracted, even without voter ID.
        # We only skip if there's absolutely no data at all (handled below).
        
        # Check if we have ANY meaningful data
        has_any_data = has_photo or has_valid_voter_id or (voter_id_text and len(voter_id_text.strip()) > 0)
        
        if has_any_data:
            should_skip = False
            skip_reason = ""
        else:
            should_skip = True
            skip_reason = "No data found (no photo, no voter ID)"

        # Log the final decision
        if should_skip:
            print(f"      ⏭️  SKIP Page {page_num+1}, Row {row+1}, Col {col+1}: {skip_reason}")
        else:
            voter_id_display = voter_id_text[:20] if voter_id_text else "[No ID]"
            photo_status = "Yes" if has_photo else "No"
            print(f"      ✅ EXTRACT Page {page_num+1}, Row {row+1}, Col {col+1}: VoterID='{voter_id_display}', Photo={photo_status}")
        
        if should_skip:
            return {'skipped': True, 'stats': cell_stats}
        
        # === EXTRACT FULL MARATHI TEXT ===
        # Try to get text from PDF text layer first (for digital PDFs)
        full_text = ""
        text_method = "none"
        is_marathi = False
        
        try:
            # Define full cell rect
            cell_full_rect = fitz.Rect(
                cell_x, cell_y, 
                cell_x + cell_width_actual, 
                cell_y + cell_height_actual
            )
            
            # 1. Try PDF Text Layer
            text_layer_content = page.get_text("text", clip=cell_full_rect).strip()
            
            # Check for Devanagari (Marathi) characters (Range: \u0900-\u097F)
            # We want to see if the text layer actually contains readable Marathi
            devanagari_chars = len(DEVANAGARI_REGEX.findall(text_layer_content))
            
            if devanagari_chars > 5:
                # Digital PDF with Marathi support
                print(f"      📝 Text Layer has Marathi ({devanagari_chars} chars). Using Text Layer.")
                full_text = text_layer_content
                text_method = "pdf_text_layer"
            
            else:
                # Scanned PDF or broken text layer -> Use OCR
                print(f"      📄 No digital Marathi text found. Using Intelligent OCR...")
                
                if local_ocr_processor:
                    # OPTIMIZATION: Crop from master image
                    cell_full_crop = None
                    if master_page_img:
                        left = cell_full_rect.x0 * master_page_scale
                        top = cell_full_rect.y0 * master_page_scale
                        right = cell_full_rect.x1 * master_page_scale
                        bottom = cell_full_rect.y1 * master_page_scale
                        cell_full_crop = master_page_img.crop((left, top, right, bottom))

                    ocr_res = local_ocr_processor.extract_full_cell_text(
                        image=cell_full_crop,
                        pdf_page=None if cell_full_crop else page,
                        rect=None if cell_full_crop else cell_full_rect,
                        fast_preprocess=(performance_mode == 'fast')
                    )
                    full_text = ocr_res.get('text', '')
                    text_method = ocr_res.get('method', 'ocr_error')
                    print(f"      ✅ OCR Text Encoded: {len(full_text)} chars")
        
        except Exception as e:
            print(f"      ⚠️  Text Extraction Error: {str(e)}")
            
        # === EXTRACT ADDITIONAL FIELDS ===
        additional_fields = {}
        fields_config = cell_template.get('fields', {})
        
        if local_ocr_processor and fields_config:
            # === OPTIMIZATION: USE MASTER PAGE IMAGE CROP ===
            # Instead of rendering every small field or cell (expensive I/O), 
            # we use the master_page_img rendered once per page.
            try:
                if master_page_img:
                    # Crop the cell from the master page image
                    left = cell_full_rect.x0 * master_page_scale
                    top = cell_full_rect.y0 * master_page_scale
                    right = cell_full_rect.x1 * master_page_scale
                    bottom = cell_full_rect.y1 * master_page_scale
                    master_cell_img = master_page_img.crop((left, top, right, bottom))
                    
                    px_scale_x = master_page_scale
                    px_scale_y = master_page_scale
                else:
                    # Fallback to legacy behavior if master image missing
                    master_cell_pix = page.get_pixmap(clip=cell_full_rect, dpi=250 if performance_mode == 'fast' else 300, alpha=False)
                    master_cell_img = Image.frombytes("RGB", [master_cell_pix.width, master_cell_pix.height], master_cell_pix.samples)
                    px_scale_x = master_cell_pix.width / cell_full_rect.width
                    px_scale_y = master_cell_pix.height / cell_full_rect.height
            except Exception as e:
                print(f"      Warning: Master cell crop failed: {e}")
                master_cell_img = None

            print(f"      🔍 Extracting {len(fields_config)} configured fields...")
            for field_key, field_box in fields_config.items():
                # Skip already extracted fields
                if field_key in ['voterID', 'photo']:
                    continue
                
                try:
                    # Calculate text rect
                    f_x = field_box.get('x', 0) * scale_x
                    f_y = field_box.get('y', 0) * scale_y
                    f_w = field_box.get('width', 0) * scale_x
                    f_h = field_box.get('height', 0) * scale_y
                    
                    field_rect = fitz.Rect(
                        cell_x + f_x,
                        cell_y + f_y, 
                        cell_x + f_x + f_w, 
                        cell_y + f_y + f_h
                    )

                    # Determine OCR strategy based on field type
                    force_marathi = False
                    key_lower = field_key.lower()
                    
                    # 1. Enforce Marathi for Name, Relative Name, Gender, and Age
                    if 'name' in key_lower or 'relative' in key_lower or 'gender' in key_lower or 'age' in key_lower or 'relation' in key_lower:
                        force_marathi = True
                    
                    # 2. Enforce English/Mixed for House No
                    if 'house' in key_lower:
                        force_marathi = False
                    
                    
                    # === CRITICAL OPTIMIZATION: TRY TEXT LAYER FIRST ===
                    # If PDF has text layer, this is 100x faster than OCR
                    layer_text = ""
                    try:
                        layer_text = page.get_text("text", clip=field_rect).strip()
                    except:
                        pass
                        
                    # Decide whether to use layer text or fallback to OCR
                    # We use layer text if it contains meaningful characters
                    use_layer_text = False
                    if layer_text:
                        # Check if it has enough content (at least 2 chars)
                        if len(layer_text) >= 1: 
                             use_layer_text = True
                             
                        # For specific fields, validate content type
                        if 'serial' in key_lower and not any(c.isdigit() for c in layer_text):
                             use_layer_text = False # Serial must have digits
                        
                        # EXCEPTION: Only Name and Relative Name using Image Processing (User Request for Speed)
                        if 'name' in key_lower or 'relative' in key_lower or 'relation' in key_lower:
                             use_layer_text = False

                    if use_layer_text:
                        # SUPER FAST PATH: Skip Image Extraction entirely!
                        print(f"         > FAST PATH (Text Layer): '{layer_text[:20]}...'")
                        field_res = {
                            'text': layer_text,
                            'raw_text': layer_text,
                            'method': 'text_layer'
                        }
                    else:
                        # SLOW PATH: Image Extraction + OCR
                        # OPTIMIZATION: Use Master Page Image if available (SUPER FAST)
                        if master_page_img and master_page_scale:
                            try:
                                # Calculate crop coordinates in master_page_img pixels
                                crop_x = int(field_rect.x0 * master_page_scale)
                                crop_y = int(field_rect.y0 * master_page_scale)
                                crop_w = int(field_rect.width * master_page_scale)
                                crop_h = int(field_rect.height * master_page_scale)
                                
                                # Ensure we don't go out of bounds
                                img_w, img_h = master_page_img.size
                                crop_x = max(0, min(crop_x, img_w - 1))
                                crop_y = max(0, min(crop_y, img_h - 1))
                                crop_w = max(1, min(crop_w, img_w - crop_x))
                                crop_h = max(1, min(crop_h, img_h - crop_y))
                                
                                crop_img = master_page_img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                            except Exception as e:
                                print(f"      Warning: Master page crop failed: {e}")
                                crop_img = None
                                
                        # OPTIMIZATION: Use Master Cell Crop as fallback
                        elif master_cell_img:
                            try:
                                # Calculate relative coordinates inside the cell
                                rel_x = field_rect.x0 - cell_rect.x0
                                rel_y = field_rect.y0 - cell_rect.y0
                                rel_w = field_rect.width
                                rel_h = field_rect.height
                                
                                # Convert to Pixels
                                crop_x = int(rel_x * px_scale_x)
                                crop_y = int(rel_y * px_scale_y)
                                crop_w = int(rel_w * px_scale_x)
                                crop_h = int(rel_h * px_scale_y)
                                
                                # Crop
                                crop_img = master_cell_img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                            except Exception as e:
                                crop_img = None

                        # Use FAST PREPROCESS if we are using a digital crop
                        use_fast_pipeline = (crop_img is not None)
                        
                        field_res = local_ocr_processor.extract_full_cell_text(
                            image=crop_img,
                            pdf_page=page if not crop_img else None, # Only fallback to page render if crop failed
                            rect=field_rect if not crop_img else None,
                            force_marathi=force_marathi,
                            fast_preprocess=use_fast_pipeline
                        )
                    
                    raw_text = field_res.get('raw_text', '').strip()
                    clean_val = field_res.get('text', '').strip()
                    
                    # === SPECIAL HANDLING FOR RELATIVE NAME & RELATION TYPE ===
                    if 'relative' in key_lower:
                        # Use strictly map_relation_type for identifying the type from raw_text
                        check_text = raw_text.replace('\n', ' ')
                        # Use map_relation_type to map to H, F, M, O
                        relation_type = TranslitHelper.map_relation_type(check_text)
                        
                        # Store in additional_fields
                        additional_fields['relationType'] = relation_type
                        
                        # Cleanup Relative Name: Remove the label part (before first colon)
                        if ':' in clean_val:
                            parts = clean_val.split(':', 1)
                            if len(parts) > 1:
                                clean_val = parts[1].strip()
                        
                        # Fallback: strict cleanup if label text is merging with name
                        # Remove common prefixes from the name value itself
                        clean_val = re.sub(r'^(पतीचे|वडिलांचे|इतर|आईचे)\s*(नाव|नावं)?[:\-\s]*', '', clean_val).strip()

                    # === SPECIAL HANDLING FOR NAME & RELATIVE NAME (Remove Numbers & Asterisks) ===
                    if 'name' in key_lower or 'relative' in key_lower:
                        # 1. Targeted OCR corrections (e.g., ठोळके -> शेळके)
                        clean_val = TranslitHelper.correct_marathi_ocr(clean_val)
                        
                        # 2. Remove digits (0-9 and Marathi ०-९) and asterisks (*)
                        clean_val = re.sub(r'[0-9०-९\*]', '', clean_val).strip()

                    # === SPECIAL HANDLING FOR AGE ===
                    if 'age' in key_lower:
                        def validate_age_text(val_str):
                            if not val_str: return None
                            # Convert Marathi digits to English
                            marathi_digits = str.maketrans("०१२३४५६७८९", "0123456789")
                            val_str = val_str.translate(marathi_digits)
                            # Keep only ASCII digits
                            val_str = re.sub(r'[^0-9]', '', val_str)
                            if not val_str: return None
                            # Sanity check: Age must be <= 100
                            try:
                                age_int = int(val_str)
                                if age_int > 100: return None
                                # Optional: minimal voter age check (e.g. > 15 to allow errors close to 18)
                                if age_int < 10: return None 
                                return val_str
                            except:
                                return None

                        # 1. Try PDF Text Layer FIRST (High accuracy)
                        layer_text = page.get_text("text", clip=field_rect).strip()
                        valid_age = validate_age_text(layer_text)

                        if valid_age:
                            clean_val = valid_age
                            print(f"         > Age found in Text Layer: {clean_val}")
                        else:
                            # 2. Fallback to OCR result (already in clean_val)
                            valid_age_ocr = validate_age_text(clean_val)
                            if valid_age_ocr:
                                clean_val = valid_age_ocr
                            else:
                                # Both failed or invalid
                                print(f"         > Age validation failed for OCR text: '{clean_val}' and Layer text: '{layer_text}'")
                                clean_val = ""

                     # === SPECIAL HANDLING FOR GENDER ===
                    if 'gender' in key_lower:
                         # Keep only Marathi/Devanagari characters and remove punctuation
                         clean_val = re.sub(r'[^\w\s\u0900-\u097F]', '', clean_val).strip()

                         # FIX SPECIFIC OCR ERRORS: Replace incorrect gender extractions with correct values
                         if clean_val in ['2 पक', 'पक', '2पक', '2 प', 'प', '2', 'पक्']:
                             clean_val = "पु"  # Replace with correct Male gender (पु)
                         elif clean_val in ['स्री', 'स्त्री', 'महिला', 'स्री.', 'स्त्री.']:
                             clean_val = "स्री"  # Ensure Female gender is properly formatted
                         else:
                             # Use TranslitHelper for Gender mapping (handles 'पर' -> 'पु', etc.)
                             gender_standard = TranslitHelper.map_gender(clean_val)
                             if gender_standard == "Male": clean_val = "पु"
                             elif gender_standard == "Female": clean_val = "स्री"

                    # === SPECIAL HANDLING FOR HOUSE NO ===
                    if 'house' in key_lower:
                        # Remove colons and common labels from house number
                        clean_val = re.sub(r'^(?:HOUSE|HS|NO|NUM)[:\- .]*', '', clean_val, flags=re.IGNORECASE)
                        clean_val = re.sub(r'[:]', '', clean_val).strip()
                    # === SPECIAL HANDLING FOR SERIAL NO / ASSEMBLY NO ===
                    if any(k in key_lower for k in ['serial', 'assembly', 'ac', 'pc', 'part']):
                        # Convert Marathi digits to English
                        marathi_digits = str.maketrans("०१२३४५६७८९", "0123456789")
                        clean_val = clean_val.translate(marathi_digits)
                        
                        if any(k in key_lower for k in ['assembly', 'part', 'ac', 'pc']):
                            # === ASSEMBLY / PART NUMBER CLEANING ===
                            # Fix OCR Typos first (Aggressive)
                            clean_val = clean_val.upper()
                            # 1. Remove Labels (Include fragments like A, P, AC, PC, and common OCR artifacts like leading digits)
                            # Only strip if it's a clear label, not part of the number
                            # Handles artifacts like "4 AC", "4Assembly", etc.
                            clean_val = re.sub(r'^\s*(?:[0-9A-Z]*\s*)?(?:ASSEMBLY|PART|AC|PC|NO|NUM|A|P|ACNO)\b\s*[:.\- ]*', '', clean_val, flags=re.IGNORECASE)
                            
                            # 2. Fix OCR substitutions (O->0, I->1, S->5, etc.)
                            clean_val = clean_val.replace('O', '0').replace('D', '0').replace('Q', '0')
                            clean_val = clean_val.replace('I', '1').replace('L', '1').replace('|', '1').replace(']', '1').replace('!', '1')
                            clean_val = clean_val.replace('Z', '2')
                            clean_val = clean_val.replace('S', '5')
                            clean_val = clean_val.replace('B', '8')
                            
                            # 3. Strip Non-Allowed Chars (Keep Digits, Slash, Hyphen, Comma)
                            # User mentioned "extract comma and all number" for serial, applying safe approach here too.
                            clean_val = re.sub(r'[^0-9/,\-]', '', clean_val)
                            
                        else:
                            # === SERIAL NUMBER CLEANING ===
                            
                            # CHECK METHOD: If Text Layer, we trust the digits more and avoid aggressive S->5
                            if field_res.get('method') == 'text_layer':
                                # Simple Cleanup for Digital Text
                                # Keep digits AND COMMAS (as per user request: "1,005" should be kept as "1,005")
                                clean_val = re.sub(r'[^0-9,]', '', clean_val)
                                # No S->5 replacement needed for digital text
                            else:
                                # OCR Path - Aggressive Cleaning
                                # 1. Fix common alpha-digit OCR confusions UPPERCASE
                                clean_val = clean_val.upper()
                                
                                # STEP A: Remove common LABELS (Be cautious with single letter 'S' to avoid stripping digit '2')
                                clean_val = re.sub(r'^\s*(?:SR|SL|NO|NUM|SERIAL|SER)\b\s*[:.\- ]*', '', clean_val, flags=re.IGNORECASE)
                                # Only strip single 'S' if followed by space or separator
                                clean_val = re.sub(r'^\s*S\s+[:.\- ]*', '', clean_val, flags=re.IGNORECASE)
                                clean_val = clean_val.strip()
    
                                # Aggressive replacements for Serial Number field (strictly numeric)
                                clean_val = clean_val.replace('O', '0').replace('D', '0').replace('Q', '0')
                                clean_val = clean_val.replace('I', '1').replace('L', '1').replace('|', '1').replace(']', '1').replace('!', '1').replace('J', '1').replace('T', '1')
                                clean_val = clean_val.replace('Z', '2')
                                clean_val = clean_val.replace('E', '3') # Common confusion
                                clean_val = clean_val.replace('A', '4')
                                
                                # Only replace S->5 if NOT at start? No, we stripped labels.
                                # But what if "S" is left? e.g. "S 56" -> "S 56" -> "5 56".
                                # Safe to replace now.
                                clean_val = clean_val.replace('S', '5') 
                                clean_val = clean_val.replace('G', '6')
                                clean_val = clean_val.replace('B', '8')
                            
                            # 2. Extract VALID number sequence (Removed strict length limits as per user request "no limitation")
                            # Find all sequences of digits (AND COMMAS) to preserve "1,005"
                            numbers = NUMBER_CLEANUP_REGEX.findall(clean_val)
                            
                            valid_serial = ""
                            if numbers:
                                # Prioritize the one that is mostly digits
                                for num in numbers:
                                    # Must contain at least one digit
                                    if any(c.isdigit() for c in num):
                                        valid_serial = num
                                        break
                                
                                if not valid_serial and numbers:
                                     valid_serial = numbers[0]
                                         
                            clean_val = valid_serial
                        
                        # Store cleaned value
                        additional_fields[field_key] = clean_val

                    additional_fields[field_key] = clean_val
                    
                    # === STANDARDIZE KEYS AND ADD ENGLISH VERSIONS ===
                    # Map common variations to standard keys expected by Excel generator
                    if 'name' in key_lower and 'relative' not in key_lower:
                        additional_fields['name'] = clean_val  # Standard key
                        additional_fields['nameEnglish'] = TranslitHelper.transliterate_marathi_to_english(clean_val)
                    elif 'relative' in key_lower and 'name' in key_lower:
                        additional_fields['relativeName'] = clean_val # Standard key
                        additional_fields['relativeNameEnglish'] = TranslitHelper.transliterate_marathi_to_english(clean_val)
                    elif 'gender' in key_lower:
                         # Use TranslitHelper for mapping to English (Male/Female)
                         # We already mapped clean_val to 'पु'/'स्री' above, but map_gender handles that too.
                         additional_fields['gender'] = clean_val # Standard key
                         additional_fields['genderEnglish'] = TranslitHelper.map_gender(clean_val)
                    elif 'age' in key_lower:
                        additional_fields['age'] = clean_val # Standard key
                    elif 'relation' in key_lower and 'type' in key_lower:
                        # Ensure the extracted Relation Type is strictly mapped using TranslitHelper
                        input_val = additional_fields.get('relationType', clean_val)
                        # Now we keep the Marathi label like 'वडिलांचे' or 'पतीचे'
                        # Ensure the extracted Relation Type is strictly mapped to codes
                        input_val = additional_fields.get('relationType', clean_val)
                        additional_fields['relationType'] = TranslitHelper.map_relation_type(input_val)
                    elif 'serial' in key_lower:
                        additional_fields['serialNo'] = clean_val # Standard key
                    elif 'house' in key_lower:
                        additional_fields['houseNo'] = clean_val # Standard key
                    elif any(k in key_lower for k in ['assembly', 'ac']):
                        additional_fields['assemblyNo'] = clean_val # Standard key
                    elif any(k in key_lower for k in ['part', 'pc']):
                        additional_fields['partNo'] = clean_val # Standard key
                    
                    print(f"         > {field_key}: '{clean_val}' (Rel: {additional_fields.get('relationType', 'N/A')})")
                    
                except Exception as ex:
                    print(f"         > {field_key}: Error ({str(ex)})")
                    additional_fields[field_key] = ""
        
        # === SMART FALLBACK (MISALIGNED GRID PROTECTION) ===
        # If critical fields are missing but full_text exists, parse from full_text
        if full_text and (not additional_fields.get('name') or len(additional_fields.get('name', '')) < 2):
            lines = [l.strip() for l in full_text.split('\n') if l.strip()]
            for line in lines:
                # Own Name is usually first line or follows 'नाव' label without relative prefixes
                if any(k in line for k in ['नाव', 'नाम']) and not any(k in line for k in ['पती', 'वडिल', 'आई', 'इतर']):
                    detected = re.sub(r'^(?:नाव|नाम)[:\- .]*', '', line).strip()
                    if detected and len(detected) > 2:
                        # Clean up colons from detected name
                        detected = re.sub(r'[:]', '', detected).strip()
                        additional_fields['name'] = TranslitHelper.correct_marathi_ocr(detected)
                        additional_fields['nameEnglish'] = TranslitHelper.transliterate_marathi_to_english(additional_fields['name'])
                        break

        if full_text and (not additional_fields.get('relativeName') or len(additional_fields.get('relativeName', '')) < 2):
            rel_prefixes = ['पतीचे', 'वडिलांचे', 'आईचे', 'इतर']
            for prefix in rel_prefixes:
                pattern = f'{prefix}\\s*(?:नाव|नाम)?[:\\- .]*(.*)'
                match = re.search(pattern, full_text)
                if match:
                    detected = match.group(1).split('\n')[0].strip()
                    if detected and len(detected) > 2:
                        # Clean up colons from detected relative name
                        detected = re.sub(r'[:]', '', detected).strip()
                        additional_fields['relativeName'] = TranslitHelper.correct_marathi_ocr(detected)
                        additional_fields['relativeNameEnglish'] = TranslitHelper.transliterate_marathi_to_english(additional_fields['relativeName'])
                        break

        # === AI SMART CORRECTION (DISABLED) ===
        # if 'name' in additional_fields and 'relativeName' in additional_fields:
        #     original_name = additional_fields['name']
        #     relative_name = additional_fields['relativeName']
        #     
        #     # Apply Correction
        #     corrected_name = TranslitHelper.smart_correct_name(original_name, relative_name)
        #     
        #     if corrected_name != original_name:
        #         additional_fields['name'] = corrected_name
        #         # Update English transliteration too
        #         additional_fields['nameEnglish'] = TranslitHelper.transliterate_marathi_to_english(corrected_name)

        # Return result

        result = {
            'page': page_num + 1,
            'column': col + 1,
            'row': row + 1,
            'voterID': voter_id_text if voter_id_text else "",
            'full_text': full_text,
            'image_base64': photo_base64,
            'nameEnglish': additional_fields.get('nameEnglish', ''),
            'relativeNameEnglish': additional_fields.get('relativeNameEnglish', ''),
            'genderEnglish': additional_fields.get('genderEnglish', ''),
            'relationTypeEnglish': TranslitHelper.transliterate_relation_type(additional_fields.get('relationType', '')),
            **additional_fields,  # MERGE DYNAMIC FIELDS
            'metadata': {
                'voter_id_confidence': voter_id_confidence,
                'photo_quality': photo_quality,
                'text_method': text_method,
                'enhanced': local_photo_processor is not None,
                'has_voter_id': has_valid_voter_id,
                'has_photo': has_photo,
                'photo_only': has_photo and not has_valid_voter_id,  
                'voter_id_only': has_valid_voter_id and not has_photo  
            },
            'stats': cell_stats,
            'skipped': False
        }
        
        # doc.close() removed - handled by caller
        return result
        
    except Exception as e:
        print(f"  ERROR in worker for cell [{cell_info.get('row', '?')+1},{cell_info.get('col', '?')+1}]: {str(e)}")
        return {'skipped': True, 'error': str(e)}

def detect_grid_offset(page, config, expected_first_cell_y):
    """
    Detects if the grid is shifted vertically on this page (e.g. larger header).
    Returns y_offset (positive = shifted down).
    """
    try:
        # Strategy: Find the "Anchor Row" containing labels like "Name", "Age", "Gender"
        # The header height is variable, but the internal cell structure is constant.
        # If we find "Name" at Y=250 instead of Y=200, we know the offset is +50.
        
        # Search area: Top 40% of page
        page_h = page.rect.height
        search_rect = fitz.Rect(0, 0, page.rect.width, page_h * 0.4)
        
        words = page.get_text("words", clip=search_rect)
        
        # Filter for anchors
        # "Name" in Marathi is "नाव" or "Naav"
        # "Age" in Marathi is "वय" or "Age"
        # "Gender" is "लिंग" or "Ling"
        
        # We look for the finding the FIRST occurrence of these that looks like a grid row
        anchors_y = []
        for w in words:
            text = w[4].strip()
            # Check matches (English or Marathi)
            if 'नाव' in text or 'Name' in text or 'Age' in text or 'वय' in text or 'लिंग' in text:
                 # Check if it's not too close to top (header title might have these words?)
                 # Usually grid starts > 100px
                 if w[1] > 50:
                     anchors_y.append(w[1])
        
        if not anchors_y:
            return 0
            
        # Cluster Y values to find lines (within 5px)
        anchors_y.sort()
        
        # Find the first "cluster" of Y values. This corresponds to the first row of cells (Row 1).
        # The "Header Reference" for this config is usually where the first cell starts.
        
        # Let's say we find the "Name" label at Y=180.
        # We need to know where "Name" label SHOULD be in the static config.
        # This is hard to know exactly without the template reference.
        
        # ALTERNATE STRATEGY: Find the FIRST Horizontal Line that spans the page width
        # This marks the delimiter between Header and Grid.
        
        drawings = page.get_drawings()
        lines = []
        for p in drawings:
            r = p['rect']
            # Horizontal line, reasonable width > 300
            if r.height < 5 and r.width > 300 and r.y0 > 100 and r.y0 < page_h * 0.4:
                lines.append(r.y0)
        
        lines.sort()
        
        detected_start_y = 0
        
        if lines:
            # The last line in the top region likely separates header from data
            detected_start_y = lines[-1]
        elif anchors_y:
             # Fallback to text: Assume "Name" label is ~25px below the cell top
             detected_start_y = anchors_y[0] - 25
        else:
            return 0
            
        # Calculate Offset
        # We compare detected_start_y with `expected_first_cell_y` (from static config)
        # expected_first_cell_y is the TOP of the first cell (grid.y)
        
        offset = detected_start_y - expected_first_cell_y
        
        # Threshold: Ignore tiny shifts (<10px) to avoid jitter
        if abs(offset) < 10:
            return 0
            
        print(f"      📏 Dynamic Header Detect: Grid Start Y={detected_start_y:.1f} (Exp: {expected_first_cell_y:.1f}) -> Offset: {offset:.1f}")
        return offset
        
        dark_rows = np.where(row_means < threshold)[0]
        
        if len(dark_rows) > 0:
            # Group consecutive dark rows into "lines"
            lines_y = []
            if len(dark_rows) > 0:
                current_group = [dark_rows[0]]
                for i in range(1, len(dark_rows)):
                    if dark_rows[i] - dark_rows[i-1] <= 2: # Combine adjacent rows
                        current_group.append(dark_rows[i])
                    else:
                        # End of line group -> Take average Y
                        avg_y = sum(current_group) / len(current_group)
                        lines_y.append(avg_y)
                        current_group = [dark_rows[i]]
                
                # Add last group
                if current_group:
                    avg_y = sum(current_group) / len(current_group)
                    lines_y.append(avg_y)
            
            if lines_y:
                # Topmost line found in image coordinates
                top_line_px = lines_y[0]
                
                # Convert back to PDF coordinates
                # PDF Y = SearchMinY + (PixelY / ZoomFactor)
                top_line_pdf_y = search_min_y + (top_line_px / 2.0)
                
                # Assume this is the top grid line
                grid_y = config.get('grid', {}).get('y', 0)
                offset = top_line_pdf_y - grid_y
                
                if abs(offset) < 500:
                    print(f"      📸 Image-Align: Detected Grid Y={top_line_pdf_y:.1f} (Exp {grid_y}), Offset={offset:.1f}")
                    return offset

        return 0
        
    except Exception as e:
        print(f"      ⚠️  Auto-Align Error: {e}")
        return 0

# === GLOBAL CACHE FOR ALIGNMENT ===
ALIGNMENT_CACHE = {}

def is_cell_empty(pix_or_img, threshold=500):
    """
    Check if a cell is effectively empty (low ink density).
    Returns True if empty.
    """
    try:
        # FAST PATH: PIL extremas
        if isinstance(pix_or_img, fitz.Pixmap):
            # Convert Pixmap to PIL for fast extrema check
            img_pil = Image.frombytes("RGB", [pix_or_img.width, pix_or_img.height], pix_or_img.samples)
            extrema = img_pil.convert('L').getextrema()
            if extrema[1] - extrema[0] < 10: return True
            img = np.array(img_pil.convert('L'))
        elif isinstance(pix_or_img, Image.Image):
            extrema = pix_or_img.convert('L').getextrema()
            if extrema[1] - extrema[0] < 10: return True
            img = np.array(pix_or_img.convert('L'))
        else:
            return False
            
        # Threshold (Black text on white background)
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        non_zero = cv2.countNonZero(thresh)
        return non_zero < threshold
    except:
        return False

def detect_page_alignment(page, config, file_id=None):
    """
    Detects the 'Anchor' Y position dynamically.
    OPTIMIZATION (Option 3):
    1. Caching: Reuse header height if file_id provided.
    2. Speed: Use 150 DPI (zoom=0.5) + CV2 HoughLines.
    """
    try:
        # 1. CHECK CACHE
        grid_y = config.get('grid', {}).get('y', 0)
        
        if file_id and file_id in ALIGNMENT_CACHE:
            cached_offset = ALIGNMENT_CACHE[file_id]
            print(f"      ⚡ Cache Hit: Using cached offset {cached_offset:.1f}")
            return cached_offset

        search_region_h = max(grid_y + 300, 500)
        
        # 2. FAST CV2 LINE DETECTION (150 DPI)
        # 150 DPI is enough for layout lines. Zoom = 150/72 ~= 2.0. Let's use 1.5 (~108 DPI) or 2.0 (144 DPI)
        # User suggested 150 DPI.
        zoom = 150 / 72.0 
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(0, 0, page.rect.width, search_region_h))
        
        import cv2
        import numpy as np
        
        img_bytes = pix.tobytes("png")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Canny + HoughLinesP
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        detected_y_pdf = None
        
        if lines is not None:
            # Filter for Horizontal Lines
            h_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 5: # Horizontal
                     h_lines.append(y1)
            
            if h_lines:
                h_lines.sort()
                
                # We want the 'First Major Line' that represents the Grid Top
                # This logic depends on where the grid starts relative to the header.
                # Let's find the line closest to our EXPECTED grid_y
                
                expected_y_px = grid_y * zoom
                closest_y_px = min(h_lines, key=lambda y: abs(y - expected_y_px))
                
                # If it's reasonable match (< 500px diff scaled)
                if abs(closest_y_px - expected_y_px) < (500 * zoom):
                     detected_y_pdf = closest_y_px / zoom
                     
        if detected_y_pdf is not None:
             offset = detected_y_pdf - grid_y
             print(f"      ⚓ Smart Align (CV2): Found Grid Line at {detected_y_pdf:.1f} (Exp {grid_y}), Offset={offset:.1f}")
             
             # UPDATE CACHE
             if file_id:
                 ALIGNMENT_CACHE[file_id] = offset
             return offset

        # STRATEGY 2: Fallback to Voter ID Regex (Original Logic) but DEEPER SEARCH
        # print("      ⚠️  CV2 Lines failed, falling back to Voter ID Regex...")
        
        # INCREASED SEARCH REGION for high headers (up to 70% of page)
        deep_search_h = max(search_region_h, page.rect.height * 0.7)
        
        text_words = page.get_text("words", clip=fitz.Rect(0, 0, page.rect.width, deep_search_h))
        voter_id_candidates = []
        
        # Use pre-compiled regex patterns for better performance
        for w in text_words:
            text = w[4].strip().upper().replace(" ", "")
            # Check detailed patterns
            if EPIC_REGEX.match(text) or LOOSE_EPIC_REGEX.match(text) or re.match(r'^[A-Z]{2,3}/[0-9]+/[0-9]+', text):
                voter_id_candidates.append(w[1]) # Append Y coordinate

        if voter_id_candidates:
            voter_id_candidates.sort() # Topmost first
            
            # Use the topmost, but verify it's not "too high" (noise) or "too low" (footer) if possible.
            # Usually the first one is the top-left cell or top-middle.
            first_id_y = voter_id_candidates[0]
            
            cell_template = config.get('cellTemplate', {})
            voter_id_box = cell_template.get('voterIdBox', {})
            vid_rel_y = voter_id_box.get('y', 10) 
            
            # The EXPECTED Grid Y derived from this found ID
            # found_y = grid_y + vid_rel_y
            # So: grid_y = found_y - vid_rel_y
            
            # Calculate offset relative to configured grid_y
            # Offset = Actual_Grid_Start - Config_Grid_Start
            # Actual_Grid_Start = first_id_y - vid_rel_y
            
            actual_grid_y = first_id_y - vid_rel_y
            offset = actual_grid_y - grid_y
            
            # Allow large offsets (up to 2000px) because header can be huge
            if abs(offset) < 2000:
                print(f"      ⚓ Smart Align (Deep ID): Found ID at {first_id_y:.1f}, Offset={offset:.1f}")
                if file_id: ALIGNMENT_CACHE[file_id] = offset
                return offset

        return 0.0 # No offset found
    except Exception as e:
        print(f"      ⚠️  Alignment Error: {e}")
        return 0.0

def process_single_page_worker(task_info):
    """
    Worker to process a WHOLE page.
    Uses Smart Anchor Detection + Option 3 Optimizations.
    """
    try:
        pdf_path = task_info['pdf_path']
        page_num = task_info['page_num']
        config = task_info['config'] # Single Config
        file_id = task_info.get('file_id', 'default') # For caching
        
        # Open PDF
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # === SMART ANCHOR ALIGNMENT (WITH CACHING & CV2) ===
        y_offset = detect_page_alignment(page, config, file_id)
        
        # === GENERATE CELLS DYNAMICALLY ===
        # Apply offset to the grid definition
        grid = config.get('grid', {})
        grid_rows = grid.get('rows', 4)
        grid_cols = grid.get('columns', 3)
        grid_x = grid.get('x', 0)
        grid_y = grid.get('y', 0)
        grid_width = grid.get('width', 1500)
        grid_height = grid.get('height', 2000)
        col_positions = grid.get('colPositions')
        row_positions = grid.get('rowPositions')
        skip_header = config.get('skipHeaderHeight', 0)
        skip_footer = config.get('skipFooterHeight', 0)
        
        # Calculate cell dimensions
        cell_width = grid_width / grid_cols
        cell_height = grid_height / grid_rows
        
        extraction_limits = (skip_header, page.rect.height - skip_footer)
        
        # Scale reference
        first_cell_width = cell_width
        first_cell_height = cell_height
        if col_positions and len(col_positions) > 1:
            first_cell_width = col_positions[1] - col_positions[0]
        if row_positions and len(row_positions) > 1:
            first_cell_height = row_positions[1] - row_positions[0]

        if row_positions and len(row_positions) > 1:
            first_cell_height = row_positions[1] - row_positions[0]

        # === OPTIMIZATION: RENDER PAGE ONCE FOR ALL CV TASKS ===
        # Use a shared 150 DPI render for both Alignment and Box Detection
        # This removes the need to render the page multiple times
        det_zoom = 150 / 72.0
        det_mat = fitz.Matrix(det_zoom, det_zoom)
        # Scan slightly above detected start if available, else standard
        # For general purpose, render the whole relevant vertical strip or just the needed area
        # To be safe and fast, we just render the likely grid area.
        
        # We need a safe start Y. If we haven't detected alignment yet, we guess.
        # But wait, we need alignment first. 
        # Actually, let's use the Box Detection results to confirm alignment too?
        # NO, user wants deep search alignment.
        
        # Let's do the rendering efficiently:
        # 1. Render the "Deep Search" area first (for text) - Text search doesn't need rendering, it uses `get_text`.
        # 2. Render the "Visual Area" for Box Detection.
        
        # Since we modified the logic to use `local_box_detector` for everything, let's prepare the image ONCE.
        
        detected_cells = []
        try:
            if BOX_DETECTOR_AVAILABLE:
                # print(f"      🔍 Attempting Auto-Grid Detection for Page {page_num + 1}...")
                local_box_detector = BoxDetector()
                
                # Determine "Safe Start Y" based on config grid_y + offset (if we calculate it early)
                # But we invoke BoxDetector to FIND the grid. 
                # Let's trust the "Deep ID" alignment we added?
                # Actually, the previous code block (which is outside this replacement chunk, around line 1262)
                # calculates 'offset'. We should use it.
                # 'y_offset' variable is available from the call: y_offset = detect_page_alignment(...)
                
                detected_start_y = grid_y + y_offset
                safe_start_y = max(0, detected_start_y - 50)
                
                # Render Clip
                det_rect = fitz.Rect(0, safe_start_y, page.rect.width, page.rect.height - skip_footer)
                
                # FAST RENDER
                det_pix = page.get_pixmap(matrix=det_mat, clip=det_rect, alpha=False) # alpha=False is faster
                
                # FAST CONVERSION (Direct Buffer - No PNG encode/decode)
                # This is much faster (~10x)
                import numpy as np
                import cv2
                
                # pix.samples is a buffer of bytes
                # shape is (height, width, n)
                # we asked for alpha=False, so n=3 (RGB) usually, or 1 (Gray)
                if det_pix.n == 3:
                    det_img = np.frombuffer(det_pix.samples, dtype=np.uint8).reshape(det_pix.h, det_pix.w, 3)
                    # Convert RGB to BGR for OpenCV
                    det_img = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
                elif det_pix.n == 4: # RGBA
                    det_img = np.frombuffer(det_pix.samples, dtype=np.uint8).reshape(det_pix.h, det_pix.w, 4)
                    det_img = cv2.cvtColor(det_img, cv2.COLOR_RGBA2BGR)
                else:
                    # Fallback for grayscale or other
                    det_img = np.frombuffer(det_pix.samples, dtype=np.uint8).reshape(det_pix.h, det_pix.w, det_pix.n)
                    if det_pix.n == 1:
                        det_img = cv2.cvtColor(det_img, cv2.COLOR_GRAY2BGR)

                # Detect Boxes
                raw_boxes = local_box_detector.detect_boxes_from_cv_image(det_img)
                # print(f"      📦 Detected {len(raw_boxes)} raw boxes")

                # Scale back to PDF coordinates
                pdf_boxes = []
                for b in raw_boxes:
                    s = 1.0 / det_zoom
                    bx = b['x'] * s
                    by = (b['y'] * s) + safe_start_y # Add safe_start_y offset
                    bw = b['width'] * s
                    bh = b['height'] * s
                    
                    if (by + bh) <= (page.rect.height - skip_footer):
                         pdf_boxes.append({
                             'x': bx, 'y': by, 'width': bw, 'height': bh
                         })
                
                # Organize into Grid
                if pdf_boxes:
                    grid_info = local_box_detector.organize_into_grid(pdf_boxes)
                    
                    d_rows = grid_info.get('rows', 0)
                    d_cols = grid_info.get('columns', 0)
                    
                    # Valid Grid Criteria
                    if d_rows >= 2 and d_cols >= 2:
                        # print(f"      ✅ Auto-Grid Success: Found {d_rows}x{d_cols} grid (Page {page_num + 1})")
                        
                        detected_grid = grid_info.get('grid', [])
                        
                        for r_idx, row_list in enumerate(detected_grid):
                            for c_idx, box in enumerate(row_list):
                                final_w = box['width']
                                final_h = box['height']
                                
                                detected_cells.append({
                                     'x': box['x'], 
                                     'y': box['y'], 
                                     'width': final_w, 
                                     'height': final_h,
                                     'row': r_idx, 
                                     'col': c_idx,
                                     # AUTO-TEMPLATE ALIGNMENT:
                                     # Scale sub-fields (Name, Photo) to match the DETECTED cell size
                                     'scale_x': final_w / first_cell_width,
                                     'scale_y': final_h / first_cell_height,
                                     'first_cell_width': first_cell_width,
                                     'first_cell_height': first_cell_height
                                })
                    else:
                        print(f"      ⚠️  Auto-Grid: Found {len(pdf_boxes)} boxes but could not form a valid grid (Found {d_rows}x{d_cols})")
            else:
                 print("      ⚠️  BoxDetector not available - skipping auto-grid")

        except Exception as e:
            print(f"      ⚠️  Auto-Grid Detection Error: {e}")
            detected_cells = []

        cells_to_process = []
        
        # DECISION: Use Detected Grid vs Static Config
        if detected_cells and len(detected_cells) > 5:
             cells_to_process = detected_cells
        else:
             print("      ⚠️  Using Manual Grid Config (Fallback)")
             # Fallback to static grid generation
             for col in range(grid_cols):
                 for row in range(grid_rows):
                     if col_positions and row_positions:
                         cx = col_positions[col] if col < len(col_positions) else grid_x + (col * cell_width)
                         cy = row_positions[row] if row < len(row_positions) else grid_y + (row * cell_height)
                         cw = (col_positions[col+1] - col_positions[col]) if col+1 < len(col_positions) else (grid_x+grid_width-cx)
                         ch = (row_positions[row+1] - row_positions[row]) if row+1 < len(row_positions) else (grid_y+grid_height-cy)
                     else:
                         cx = grid_x + (col * cell_width)
                         cy = grid_y + (row * cell_height)
                         cw, ch = cell_width, cell_height
                     
                     # Apply offset (Only for Static Grid)
                     cy += y_offset
                         
                     cells_to_process.append({
                         'x': cx, 'y': cy, 'width': cw, 'height': ch,
                         'row': row, 'col': col,
                         'scale_x': cw / first_cell_width,
                         'scale_y': ch / first_cell_height,
                         'first_cell_width': first_cell_width,
                         'first_cell_height': first_cell_height
                     })

        # Initialize processors locally
        processors = {
            'ocr': None,
            'photo': None,
            'smart': None,
            'box': None
        }
        
        try:
            if OCR_400DPI_AVAILABLE:
                from ocr_processor_400dpi import OCRProcessor400DPI
                proc_ocr = OCRProcessor400DPI()
                # Set language from config if provided
                target_lang = config.get('language', 'mr')
                proc_ocr.set_ocr_language(target_lang)
                processors['ocr'] = proc_ocr
        except: pass
        
        try:
            if PHOTO_PROCESSOR_AVAILABLE:
                from photo_processor import PhotoProcessor
                processors['photo'] = PhotoProcessor()
        except: pass
        
        try:
            if SMART_DETECTOR_AVAILABLE:
                from smart_detector import SmartDetector
                processors['smart'] = SmartDetector()
        except: pass

        try:
            if BOX_DETECTOR_AVAILABLE:
                from box_detector import BoxDetector
                processors['box'] = BoxDetector()
        except: pass
        
        results = []
        
        # === PERFORMANCE OPTIMIZATION: RENDER PAGE ONCE ===
        # Use a shared render for all extraction tasks (Header + Cells)
        # Optimization: use 200 DPI in fast mode to speed up rendering and cropping
        performance_mode = config.get('performanceMode', 'balanced')
        render_dpi = 200 if performance_mode == 'fast' else 300
        master_page_img = None
        master_page_scale = render_dpi / 72.0 
        try:
            page_pix = page.get_pixmap(dpi=render_dpi, alpha=False)
            master_page_img = Image.frombytes("RGB", [page_pix.width, page_pix.height], page_pix.samples)
        except Exception as e:
            print(f"      ⚠️  Master page render failed: {e}")

        # === EXTRACT PAGE LEVEL FIELDS (Booth Info) ===
        page_data = {}
        # New constant fields from config
        page_data['prabhag'] = config.get('prabhag', '')
        page_data['boothNo'] = config.get('boothNo', '')

        page_template = config.get('pageTemplate', {})
        
        if page_template and processors.get('ocr'):
             for field_key, field_box in page_template.items():
                    try:
                        r_x = field_box.get('x', 0)
                        r_y = field_box.get('y', 0)
                        r_w = field_box.get('width', 0)
                        r_h = field_box.get('height', 0)
                        
                        full_rect = fitz.Rect(r_x, r_y, r_x + r_w, r_y + r_h)
                        
                        # Apply y_offset to header fields too if they are shifted
                        # (Usually header is static, but sometimes shifted with the grid)
                        # We use a small fraction of offset for header if it's very large
                        # But typically header elements relate to page top 
                        
                        # Crop from master image for speed and quality
                        cropped_field_img = None
                        if master_page_img:
                            left = r_x * master_page_scale
                            top = r_y * master_page_scale
                            right = (r_x + r_w) * master_page_scale
                            bottom = (r_y + r_h) * master_page_scale
                            cropped_field_img = master_page_img.crop((left, top, right, bottom))

                        # Determine if this is a booth-related field (Center or Address)
                        is_booth_field = any(k in field_key.lower() for k in ['booth', 'center', 'address'])
                        
                        # Rule: For booth fields, do NOT force Marathi-only
                        force_marathi_val = not is_booth_field
                        
                        field_res = processors['ocr'].extract_full_cell_text(
                            image=cropped_field_img,
                            pdf_page=None if cropped_field_img else page,
                            rect=None if cropped_field_img else full_rect,
                            force_marathi=force_marathi_val 
                        )
                        
                        # RULE: For booth fields, use raw text + special cleaning for Z.P.
                        if is_booth_field:
                            val = field_res.get('raw_text', '').strip()
                            val = TranslitHelper.clean_booth_info(val)
                        else:
                            val = field_res.get('text', '').strip()
                            # CLEANUP: Remove common OCR garbage for standard fields, but be careful with digits
                            val = re.sub(r'[|:;!॥\-]', '', val).strip()
                        
                        val = ' '.join(val.split())
                        
                        page_data[field_key] = val
                        
                        # Populate English variant
                        if val:
                            try:
                                # RULE: If the field already contains English characters, use it directly
                                if re.search(r'[a-zA-Z]', val):
                                    page_data[f"{field_key}English"] = val
                                else:
                                    english_val = TranslitHelper.transliterate_marathi_to_english(val)
                                    page_data[f"{field_key}English"] = english_val
                            except:
                                page_data[f"{field_key}English"] = ""
                                
                    except Exception as e:
                        print(f"      ⚠️  Header field {field_key} error: {e}")
                        pass

             # === SMART FALLBACK: Search for missing booth info if not found via template ===
             # If boothCenter or boothAddress is missing or very short, try to find it by scanning the top region
             if (not page_data.get('boothCenter') or len(page_data.get('boothCenter', '')) < 5) and master_page_img:
                 try:
                     # print(f"      🔍 Smart Fallback: Searching for Booth Center in header...")
                     # Scan Top 15% of the page
                     header_h = int(page.rect.height * 0.15)
                     header_rect = fitz.Rect(0, 0, page.rect.width, header_h)
                     
                     # Extract all text from header using OCR
                     header_pix = page.get_pixmap(clip=header_rect, dpi=200, alpha=False)
                     header_img = Image.frombytes("RGB", [header_pix.width, header_pix.height], header_pix.samples)
                     
                     header_res = processors['ocr'].extract_full_cell_text(
                         image=header_img,
                         force_marathi=False
                     )
                     
                     header_text = header_res.get('raw_text', '')
                     if header_text:
                         # Look for Booth Center keywords: "मतदान केंद्राचे नाव", "Polling Station Name"
                         # Logic: Usually the text AFTER the label is the value
                         booth_patterns = [
                             r'(?:मतदान केंद्राचे नाव|नाम|नाव)\s*[:\-]*\s*(.*)',
                             r'(?:Polling Station Name|Station Name)\s*[:\-]*\s*(.*)'
                         ]
                         
                         for pattern in booth_patterns:
                             match = re.search(pattern, header_text, re.IGNORECASE)
                             if match:
                                 detected_val = match.group(1).split('\n')[0].strip()
                                 if len(detected_val) > 5:
                                     # print(f"      ✨ Smart Detected Booth Center: {detected_val}")
                                     page_data['boothCenter'] = TranslitHelper.clean_booth_info(detected_val)
                                     try:
                                         if re.search(r'[a-zA-Z]', detected_val):
                                             page_data['boothCenterEnglish'] = detected_val
                                         else:
                                             page_data['boothCenterEnglish'] = TranslitHelper.transliterate_marathi_to_english(detected_val)
                                     except: pass
                                     break
                         
                         # Look for Booth Address keywords: "पत्ता", "Address"
                         address_patterns = [
                             r'(?:मतदान केंद्राचे पत्ता|पत्ता)\s*[:\-]*\s*(.*)',
                             r'(?:Polling Station Address|Address)\s*[:\-]*\s*(.*)'
                         ]
                         
                         for pattern in address_patterns:
                             match = re.search(pattern, header_text, re.IGNORECASE)
                             if match:
                                 detected_val = match.group(1).split('\n')[0].strip()
                                 if len(detected_val) > 5:
                                     # print(f"      ✨ Smart Detected Booth Address: {detected_val}")
                                     page_data['boothAddress'] = TranslitHelper.clean_booth_info(detected_val)
                                     try:
                                         if re.search(r'[a-zA-Z]', detected_val):
                                             page_data['boothAddressEnglish'] = detected_val
                                         else:
                                             page_data['boothAddressEnglish'] = TranslitHelper.transliterate_marathi_to_english(detected_val)
                                     except: pass
                                     break
                 except: pass

        # Process Cells
        for cell_info in cells_to_process:
            result = _extract_cell_internal(
                page=page,
                page_num=page_num,
                cell_info=cell_info,
                config=config, 
                extraction_limits=extraction_limits, 
                processors=processors,
                master_page_img=master_page_img,
                master_page_scale=master_page_scale
            )
            
            if result and not result.get('skipped'):
                for pk, pv in page_data.items():
                    result[pk] = pv
                
                # Transliterate Cell Fields (copy existing logic)
                try:
                    name_marathi = result.get('name', '')
                    if name_marathi:
                        result['nameEnglish'] = TranslitHelper.transliterate_marathi_to_english(name_marathi)
                    
                    rel_name_marathi = result.get('relativeName', '')
                    if rel_name_marathi:
                        result['relativeNameEnglish'] = TranslitHelper.transliterate_marathi_to_english(rel_name_marathi)
                except:
                    pass
            
            results.append(result)
            
        doc.close()
        return results
        
    except Exception as e:
        print(f"CRITICAL ERROR in page worker {task_info.get('page_num')}: {e}")
        import traceback
        traceback.print_exc()
        return []

# GLOBAL WORKER CACHE
WORKER_OCR = None
WORKER_BOX = None

def init_worker():
    """
    Initializer for Multiprocessing Workers.
    Creates the OCR Processor AND Box Detector ONCE per process.
    """
    global WORKER_OCR, WORKER_BOX
    import os
    pid = os.getpid()
    try:
        # print(f"   🔧 Worker {pid}: Initializing Processors...")
        from ocr_processor_400dpi import OCRProcessor400DPI
        WORKER_OCR = OCRProcessor400DPI() 
        
        # Initialize BoxDetector if module available
        if BOX_DETECTOR_AVAILABLE:
            from box_detector import BoxDetector
            WORKER_BOX = BoxDetector()
            
        # print(f"   ✅ Worker {pid}: Ready.")
    except Exception as e:
        print(f"   ❌ Worker {pid} Init Failed: {e}")

def process_page(pdf_path, page_num, config, template, box_detector_config=None):
    """
    Process a single page (Worker Function).
    """
    try:
        import os
        pid = os.getpid()
        
        # Use Cached Processor
        global WORKER_OCR, WORKER_BOX
        
        if WORKER_OCR is None:
             # Fallback
             from ocr_processor_400dpi import OCRProcessor400DPI
             WORKER_OCR = OCRProcessor400DPI()
             
        local_ocr_processor = WORKER_OCR
        # Update quality mode based on config
        performance_mode = config.get('performanceMode', 'balanced')
        local_ocr_processor.set_quality_mode(performance_mode)
        
        # Use Cached Box Detector
        local_box_detector = None
        # FORCE ENABLE: If BoxDetector is available, we USE it for Dynamic Grid (User Request)
        # Verify if we should use it? User asked to "enable" it.
        # We'll treat it as enabled by default if the module is loaded.
        use_auto_grid = BOX_DETECTOR_AVAILABLE
        
        if use_auto_grid:
            if WORKER_BOX is None:
                from box_detector import BoxDetector
                WORKER_BOX = BoxDetector()
            local_box_detector = WORKER_BOX
        
        # Open PDF (Must be done in worker)
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # === SMART ANCHOR ALIGNMENT (WITH CACHING & CV2) ===
        # This part needs to be re-evaluated for worker context.
        # For now, assume alignment is done once per page.
        # If detect_page_alignment uses CV2, it might need its own processor.
        # For simplicity, let's assume it's lightweight or already cached.
        y_offset = detect_page_alignment(page, config, f"{pdf_path}_{page_num}") # Pass unique file_id
        detected_cells = []
        
        # === GENERATE CELLS DYNAMICALLY ===
        grid = config.get('grid', {})
        grid_rows = grid.get('rows', 4)
        grid_cols = grid.get('columns', 3)
        grid_x = grid.get('x', 0)
        grid_y = grid.get('y', 0)
        grid_width = grid.get('width', 1500)
        grid_height = grid.get('height', 2000)
        col_positions = grid.get('colPositions')
        row_positions = grid.get('rowPositions')
        skip_header = config.get('skipHeaderHeight', 0)
        skip_footer = config.get('skipFooterHeight', 0)
        
        cell_width = grid_width / grid_cols
        cell_height = grid_height / grid_rows
        
        extraction_limits = (skip_header, page.rect.height - skip_footer)
        
        first_cell_width = cell_width
        first_cell_height = cell_height
        if col_positions and len(col_positions) > 1:
            first_cell_width = col_positions[1] - col_positions[0]
        if row_positions and len(row_positions) > 1:
            first_cell_height = row_positions[1] - row_positions[0]

        # Box detection logic
        if use_auto_grid and local_box_detector:
            try:
                # Use cached local_box_detector (Already initialized in worker header)
                # ...
                
                detected_start_y = grid_y + y_offset
                safe_start_y = max(0, detected_start_y - 50)
                
                det_zoom = 150 / 72.0
                det_mat = fitz.Matrix(det_zoom, det_zoom)
                det_rect = fitz.Rect(0, safe_start_y, page.rect.width, page.rect.height - skip_footer)
                det_pix = page.get_pixmap(matrix=det_mat, clip=det_rect, alpha=False)
                
                import numpy as np
                import cv2
                if det_pix.n == 3:
                    det_img = np.frombuffer(det_pix.samples, dtype=np.uint8).reshape(det_pix.h, det_pix.w, 3)
                    det_img = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
                elif det_pix.n == 4:
                    det_img = np.frombuffer(det_pix.samples, dtype=np.uint8).reshape(det_pix.h, det_pix.w, 4)
                    det_img = cv2.cvtColor(det_img, cv2.COLOR_RGBA2BGR)
                else:
                    det_img = np.frombuffer(det_pix.samples, dtype=np.uint8).reshape(det_pix.h, det_pix.w, det_pix.n)
                    if det_pix.n == 1:
                        det_img = cv2.cvtColor(det_img, cv2.COLOR_GRAY2BGR)

                raw_boxes = local_box_detector.detect_boxes_from_cv_image(det_img)
                
                pdf_boxes = []
                for b in raw_boxes:
                    s = 1.0 / det_zoom
                    bx = b['x'] * s
                    by = (b['y'] * s) + safe_start_y
                    bw = b['width'] * s
                    bh = b['height'] * s
                    if (by + bh) <= (page.rect.height - skip_footer):
                         pdf_boxes.append({'x': bx, 'y': by, 'width': bw, 'height': bh})
                
                if pdf_boxes:
                    grid_info = local_box_detector.organize_into_grid(pdf_boxes)
                    d_rows = grid_info.get('rows', 0)
                    d_cols = grid_info.get('columns', 0)
                    if d_rows >= 2 and d_cols >= 2:
                        detected_grid = grid_info.get('grid', [])
                        for r_idx, row_list in enumerate(detected_grid):
                            for c_idx, box in enumerate(row_list):
                                final_w = box['width']
                                final_h = box['height']
                                detected_cells.append({
                                     'x': box['x'], 'y': box['y'], 'width': final_w, 'height': final_h,
                                     'row': r_idx, 'col': c_idx,
                                     'scale_x': final_w / first_cell_width, 'scale_y': final_h / first_cell_height,
                                     'first_cell_width': first_cell_width, 'first_cell_height': first_cell_height
                                })
            except Exception as e:
                print(f"      ⚠️  Auto-Grid Detection Error in worker for page {page_num+1}: {e}")

        cells_to_process = []
        if detected_cells and len(detected_cells) > 5:
             cells_to_process = detected_cells
        else:
             for col in range(grid_cols):
                 for row in range(grid_rows):
                     if col_positions and row_positions:
                         cx = col_positions[col] if col < len(col_positions) else grid_x + (col * cell_width)
                         cy = row_positions[row] if row < len(row_positions) else grid_y + (row * cell_height)
                         cw = (col_positions[col+1] - col_positions[col]) if col+1 < len(col_positions) else (grid_x+grid_width-cx)
                         ch = (row_positions[row+1] - row_positions[row]) if row+1 < len(row_positions) else (grid_y+grid_height-cy)
                     else:
                         cx = grid_x + (col * cell_width)
                         cy = grid_y + (row * cell_height)
                         cw, ch = cell_width, cell_height
                     
                     cy += y_offset
                         
                     cells_to_process.append({
                         'x': cx, 'y': cy, 'width': cw, 'height': ch,
                         'row': row, 'col': col,
                         'scale_x': cw / first_cell_width,
                         'scale_y': ch / first_cell_height,
                         'first_cell_width': first_cell_width,
                         'first_cell_height': first_cell_height
                     })

        # Initialize other processors locally if needed, or pass them.
        # For now, only OCR is globally cached.
        processors = {
            'ocr': local_ocr_processor,
            'photo': None,
            'smart': None,
            'box': None
        }
        
        try:
            if PHOTO_PROCESSOR_AVAILABLE:
                from photo_processor import PhotoProcessor
                processors['photo'] = PhotoProcessor()
        except: pass
        
        try:
            if SMART_DETECTOR_AVAILABLE:
                from smart_detector import SmartDetector
                processors['smart'] = SmartDetector()
        except: pass

        # BoxDetector is already used above for grid detection, if needed.
        # If it's needed for _extract_cell_internal, it should be initialized here.
        # For now, let's assume it's not needed or handled by the main process.
        # if BOX_DETECTOR_AVAILABLE:
        #     processors['box'] = BoxDetector()
        
        page_results = []
        
        # === PERFORMANCE OPTIMIZATION: RENDER PAGE ONCE ===
        # Choose DPI based on performance mode
        performance_mode = config.get('performanceMode', 'balanced')
        if performance_mode == 'fast':
            render_dpi = 250
        elif performance_mode == 'balanced':
            render_dpi = 300
        else:
            render_dpi = 400
            
        master_page_img = None
        master_page_scale = render_dpi / 72.0
        try:
            page_pix = page.get_pixmap(dpi=render_dpi, alpha=False)
            master_page_img = Image.frombytes("RGB", [page_pix.width, page_pix.height], page_pix.samples)
        except Exception as e:
            print(f"      ⚠️  Master page render failed in worker: {e}")

        # === EXTRACT PAGE LEVEL FIELDS (Booth Info) ===
        page_data = {}
        # New constant fields from config
        page_data['prabhag'] = config.get('prabhag', '')
        page_data['boothNo'] = config.get('boothNo', '')
        
        page_template = config.get('pageTemplate', {})
        
        if page_template and processors.get('ocr'):
             for field_key, field_box in page_template.items():
                    try:
                        r_x = field_box.get('x', 0)
                        r_y = field_box.get('y', 0)
                        r_w = field_box.get('width', 0)
                        r_h = field_box.get('height', 0)
                        
                        full_rect = fitz.Rect(r_x, r_y, r_x + r_w, r_y + r_h)

                        # Crop from master image
                        cropped_field_img = None
                        if master_page_img:
                            left = r_x * master_page_scale
                            top = r_y * master_page_scale
                            right = (r_x + r_w) * master_page_scale
                            bottom = (r_y + r_h) * master_page_scale
                            cropped_field_img = master_page_img.crop((left, top, right, bottom))
                        
                        # Determine if this is a booth-related field
                        is_booth_field = any(k in field_key.lower() for k in ['booth', 'center', 'address'])
                        force_marathi_val = not is_booth_field
                        
                        field_res = processors['ocr'].extract_full_cell_text(
                            image=cropped_field_img,
                            pdf_page=None if cropped_field_img else page,
                            rect=None if cropped_field_img else full_rect,
                            force_marathi=force_marathi_val 
                        )
                        
                        # RULE: For booth fields, use raw text + special cleaning for Z.P.
                        if is_booth_field:
                            val = field_res.get('raw_text', '').strip()
                            val = TranslitHelper.clean_booth_info(val)
                        else:
                            val = field_res.get('text', '').strip()
                            # CLEANUP: Remove common OCR garbage for standard fields
                            val = re.sub(r'[|:;!॥\--]', '', val).strip()
                            
                        val = ' '.join(val.split())
                        
                        page_data[field_key] = val
                        
                        if val:
                            try:
                                # RULE: Transliterate to English if Devanagari is present
                                if any('\u0900' <= char <= '\u097F' for char in val):
                                     english_val = TranslitHelper.transliterate_marathi_to_english(val)
                                     page_data[f"{field_key}English"] = english_val
                                else:
                                     page_data[f"{field_key}English"] = val
                            except:
                                page_data[f"{field_key}English"] = ""
                    except Exception as e:
                        pass

             # === SMART FALLBACK: Search for missing booth info if not found via template ===
             if (not page_data.get('boothCenter') or len(page_data.get('boothCenter', '')) < 5) and master_page_img:
                 try:
                     # OPTIMIZATION: Use master image instead of rendering again
                     header_h = int(page.rect.height * 0.15)
                     header_img = master_page_img.crop((0, 0, int(page.rect.width * master_page_scale), int(header_h * master_page_scale)))
                     
                     header_res = processors['ocr'].extract_full_cell_text(image=header_img, force_marathi=False)
                     header_text = header_res.get('raw_text', '')
                     if header_text:
                         # Booth Center patterns
                         for p in [r'(?:मतदान केंद्राचे नाव|नाम|नाव)\s*[:\-]*\s*(.*)', r'(?:Polling Station Name|Station Name)\s*[:\-]*\s*(.*)']:
                             m = re.search(p, header_text, re.IGNORECASE)
                             if m:
                                 det = m.group(1).split('\n')[0].strip()
                                 if len(det) > 5:
                                     # FIX: Common Z.P. OCR error
                                     det = re.sub(r'\b2[,. ]+2\b', 'Z.P.', det)
                                     det = re.sub(r'\b2[,. ]+P\b', 'Z.P.', det)
                                     det = TranslitHelper.clean_booth_info(det); page_data['boothCenter'] = det
                                     try: 
                                         # RULE: Transliterate to English if Devanagari is present
                                         if any('\u0900' <= char <= '\u097F' for char in det):
                                             page_data['boothCenterEnglish'] = TranslitHelper.transliterate_marathi_to_english(det)
                                         else:
                                             page_data['boothCenterEnglish'] = det
                                     except: pass
                                     break
                         # Booth Address patterns
                         for p in [r'(?:मतदान केंद्राचे पत्ता|पत्ता)\s*[:\-]*\s*(.*)', r'(?:Polling Station Address|Address)\s*[:\-]*\s*(.*)']:
                             m = re.search(p, header_text, re.IGNORECASE)
                             if m:
                                 det = m.group(1).split('\n')[0].strip()
                                 if len(det) > 5:
                                     # FIX: Common Z.P. OCR error
                                     det = re.sub(r'\b2[,. ]+2\b', 'Z.P.', det)
                                     det = re.sub(r'\b2[,. ]+P\b', 'Z.P.', det)
                                     det = TranslitHelper.clean_booth_info(det); page_data['boothAddress'] = det
                                     try: 
                                         # RULE: Transliterate to English if Devanagari is present
                                         if any('\u0900' <= char <= '\u097F' for char in det):
                                             page_data['boothAddressEnglish'] = TranslitHelper.transliterate_marathi_to_english(det)
                                         else:
                                             page_data['boothAddressEnglish'] = det
                                     except: pass
                                     break
                 except: pass

        # Process Cells
        for cell_info in cells_to_process:
            result = _extract_cell_internal(
                page=page,
                page_num=page_num,
                cell_info=cell_info,
                config=config,
                extraction_limits=extraction_limits, 
                processors=processors,
                master_page_img=master_page_img,
                master_page_scale=master_page_scale
            )
            
            if result and not result.get('skipped'):
                for pk, pv in page_data.items():
                    result[pk] = pv
                
                try:
                    name_marathi = result.get('name', '')
                    if name_marathi:
                        result['nameEnglish'] = TranslitHelper.transliterate_marathi_to_english(name_marathi)
                    
                    rel_name_marathi = result.get('relativeName', '')
                    if rel_name_marathi:
                        result['relativeNameEnglish'] = TranslitHelper.transliterate_marathi_to_english(rel_name_marathi)
                except:
                    pass
            
            page_results.append(result)
            
        doc.close()
        return page_results
        
    except Exception as e:
        print(f"CRITICAL ERROR in page worker {page_num}: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_grid_vertical_enhanced(pdf_bytes, config, pdf_path=None):
    """
    Enhanced extraction with local OCR - OPTIMIZED FOR LARGE FILES
    Refactored to process by PAGE instead of by CELL to reduce overhead.
    """
    import time
    import tempfile
    import multiprocessing as mp
    
    # Start timing
    start_time = time.time()
    
    print("=" * 60)
    print("ENHANCED EXTRACTION (Optimized) - Page Level Parallelism")
    print("=" * 60)
    
    # Ensure we have a file path (required for efficient multiprocessing)
    temp_file = None
    if not pdf_path:
        print("WARNING: No PDF path provided, creating temp file for workers...")
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        with os.fdopen(temp_fd, 'wb') as f:
            f.write(pdf_bytes)
        pdf_path = temp_path
        temp_file = temp_path
    
    try:
        # Open PDF to plan tasks
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"PDF opened: {total_pages} pages")
        
        # Calculate valid page range
        skip_start = config.get('skipPagesStart', 0)
        skip_end = config.get('skipPagesEnd', 0)
        start_page = skip_start
        end_page = total_pages - skip_end
        
        print(f"Processing pages {start_page + 1} to {end_page}")
        
        # Prepare Config
        worker_config = {
            'grid': config.get('grid', {}),
            'cellTemplate': config.get('cellTemplate', {}),
            'pageTemplate': config.get('pageTemplate', {}),
            'skipHeaderHeight': config.get('skipHeaderHeight', 0),
            'skipFooterHeight': config.get('skipFooterHeight', 0),
            'prabhag': config.get('prabhag', '') or config.get('Prabhag', ''), # Handle case variants
            'boothNo': config.get('boothNo', '') or config.get('BoothNo', ''), # Handle case variants
            'performanceMode': config.get('performanceMode', 'balanced')
        }
        
        print(f"      DEBUG: Passing Prabhag='{worker_config['prabhag']}', BoothNo='{worker_config['boothNo']}' to workers")
        
        # PREPARE TASKS BY PAGE
        work_items = []
        
        for page_num in range(start_page, end_page):
            # Create Task for process_page function
            work_items.append((
                pdf_path,
                page_num,
                worker_config,
                config.get('cellTemplate', {}), # template argument for process_page
                config.get('boxDetectorConfig') # Pass box detector config if available
            ))
            
        print(f"\nCreated {len(work_items)} page tasks (OPTIMIZED)")
        print(f"Starting parallel processing with {CPU_WORKERS} workers...\n")
        
        # EXECUTE PARALLEL
        parallel_start = time.time()
        results_flat = []
        
        max_workers = CPU_WORKERS
        if max_workers > 1:
            try:
                # Pass initializer to create OCR Processor once per worker
                with mp.Pool(processes=max_workers, initializer=init_worker) as pool:
                    # We use starmap to unpack arguments
                    # process_page needs to be top-level pickleable. It is.
                    map_results = pool.starmap(process_page, work_items)
                    
                    # Flatten results (each process_page returns a list of voters)
                    for res_list in map_results:
                        if res_list:
                            results_flat.extend(res_list)
            except Exception as e:
                print(f"Parallel execution failed: {e}")
        else:
             results_flat = [item for t in page_tasks for item in process_single_page_worker(t)]

        print(f"Parallel time: {time.time() - parallel_start:.2f}s")
        
        # AGGREGATE STATS
        extracted_data = []
        stats = {'total_cells': 0, 'cells_skipped': 0}

        # Memory optimization: Process results in batches to reduce memory usage
        batch_size = 1000
        for i, res in enumerate(results_flat):
            if not res: continue
            stats['total_cells'] += 1

            if res.get('skipped', False):
                stats['cells_skipped'] += 1
                # Aggregate sub-stats
                for k,v in res.get('stats', {}).items():
                     stats[k] = stats.get(k, 0) + v
            else:
                extracted_data.append(res)
                for k,v in res.get('stats', {}).items():
                     stats[k] = stats.get(k, 0) + v

            # Memory optimization: Force garbage collection periodically
            if i % batch_size == 0 and i > 0:
                gc.collect()
        
        # Sort results by Serial No primarily, fallback to physical order
        def get_sort_key(x):
            s = x.get('serialNo')
            if s and isinstance(s, str):
                # Faster numeric extraction
                digits = "".join(filter(str.isdigit, s))
                if digits: return (0, int(digits))
            return (1, x.get('page', 0), x.get('column', 0), x.get('row', 0))

        extracted_data.sort(key=get_sort_key)
        
        total_time = time.time() - start_time
        print(f"Total Time: {total_time:.2f}s")
        print(f"Extracted: {len(extracted_data)}/{stats['total_cells']}")
        
        return {
            'extracted_data': extracted_data,
            'stats': {
                'records_extracted': len(extracted_data),
                'cells_processed': stats['total_cells'],
                'cells_skipped': stats['cells_skipped'],
                'extraction_time_seconds': round(total_time, 2)
            }
        }
        
    finally:
        # Cleanup temp file if we created one
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except: pass


def extract_grid_vertical(pdf_bytes, config, pdf_path=None):
    """
    Main extraction function - uses enhanced version with local OCR
    
    Returns:
        If enhanced version returns dict with stats, extract just the data list
        Otherwise returns the list directly for backward compatibility
    """
    result = extract_grid_vertical_enhanced(pdf_bytes, config, pdf_path=pdf_path)
    
    # If result is a dict with stats, return it as-is (new format)
    if isinstance(result, dict) and 'extracted_data' in result:
        return result
    
    # Otherwise return result directly (backward compatibility)
    return result


def clean_voter_id(text):
    """
    Clean and normalize voter ID text (fallback method)
    
    Args:
        text: Raw OCR text
    
    Returns:
        Cleaned voter ID text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common OCR errors
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Try to extract voter ID pattern (e.g., NOW1234567)
    # Common patterns: 3 letters followed by 7 digits
    pattern = r'[A-Z]{3}\d{7}'
    match = re.search(pattern, text.upper())
    if match:
        cleaned = match.group(0)
    else:
        # If no pattern match, return cleaned text
        cleaned = text.strip()
    
    # Remove trailing underscores (common OCR error)
    cleaned = cleaned.rstrip('_').strip()
    
    return cleaned


def test_tesseract():
    """
    Test if Tesseract is properly installed
    """
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        
        # Check available languages
        langs = pytesseract.get_languages()
        print(f"Available languages: {langs}")
        
        if 'eng' not in langs:
            print("WARNING: English language data not found")
        if 'hin' not in langs:
            print("WARNING: Hindi language data not found")
        
        return True
    except Exception as e:
        print(f"Tesseract test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test enhanced extraction modules
    print("Testing Enhanced Extraction Modules...")
    print("=" * 60)
    
    # Test Tesseract installation
    print("\n1. Testing Tesseract OCR (fallback)...")
    test_tesseract()
    
    # Test advanced modules
    print("\n2. Testing Advanced Modules...")
    
    if ocr_processor_400dpi:
        print("  OK: 400 DPI OCR Processor: Available ✓")
    else:
        print("  FAIL: 400 DPI OCR Processor: Not available")
    
    if photo_processor:
        print("  OK: Photo Processor: Available")
    else:
        print("  FAIL: Photo Processor: Not available")
    
    if box_detector:
        print("  OK: Box Detector: Available")
    else:
        print("  FAIL: Box Detector: Not available")
    
    if smart_detector:
        print("  OK: Smart Detector: Available")
    else:
        print("  FAIL: Smart Detector: Not available")
    
    print("\n" + "=" * 60)
    print("Ready for enhanced extraction!")
    if ocr_processor_400dpi:
        print("Strategy: 400 DPI Local OCR ✓")
    else:
        print("Strategy: Standard Tesseract OCR")
    print("=" * 60)