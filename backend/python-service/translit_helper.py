import re
import io
import os
import base64
from deep_translator import GoogleTranslator

class TranslitHelper:
    """
    Helper class for English-only voter data extraction.
    Removed all Marathi-specific logic as requested.
    """
    
    _translit_cache = {}
    _kannada_translator = GoogleTranslator(source='en', target='kn')

    @staticmethod
    def correct_ocr_misreads(text):
        """
        Fix common English OCR misreads for voter documents.
        This should be used carefully to avoid mangling names.
        """
        if not text: return ""
        
        # 1. Fix common gender misreads (more robust)
        # Only if the text looks like it might be a gender field
        t_upper = text.strip().upper()
        if t_upper in ['NALE', 'PIALE', 'PU', 'PUR', 'MALE']:
             return "Male"
        if t_upper in ['RENALE', 'STRI', 'MAHILA', 'FEM', 'FE', 'FEMALE']:
             return "Female"

        # 2. Fix common "Name" label misreads (pre-cleaning)
        text = re.sub(r'\b(Nam|Nav|Nam|Nav|Nam)\b', 'Name', text, flags=re.IGNORECASE)
        
        return text

    @staticmethod
    def transliterate_marathi_to_english(text):
        """
        No longer transliterating from Marathi.
        Just cleaning and returning English text as requested.
        """
        if not text: return ""
        text = TranslitHelper.correct_ocr_misreads(text)
        # Standardize capitalization for voter data
        return ' '.join(word.capitalize() for word in text.split()).strip()

    @staticmethod
    def transliterate_marathi_to_kannada(text):
        """
        Since input is now English-only, we translate English to Kannada.
        """
        return TranslitHelper.translate_to_kannada(text)

    @staticmethod
    def translate_to_kannada(text):
        """
        Translate English text to Kannada using Google Translator.
        """
        if not text or len(text.strip()) < 2:
            return ""
        try:
            # Check cache
            cache_key = f"en_kn_{text}"
            if cache_key in TranslitHelper._translit_cache:
                return TranslitHelper._translit_cache[cache_key]
            
            result = TranslitHelper._kannada_translator.translate(text)
            TranslitHelper._translit_cache[cache_key] = result
            return result
        except:
            return ""

    @staticmethod
    def map_gender(text):
        if not text:
            return ""
        t = str(text).strip().upper()
        
        # 1. Direct priority matches
        if any(k in t for k in ['FEMALE', 'STRI', 'RENALE', 'FE ']):
             return "Female"
        if any(k in t for k in ['MALE', 'PURUSH', 'NALE', 'PIALE']):
             return "Male"
             
        # 2. Strict character matches
        if t in ['M', 'M.', 'MALE']: return "Male"
        if t in ['F', 'F.', 'FEMALE']: return "Female"

        # 3. Fuzzy matches for common OCR misreads
        if t.startswith('M') or t.endswith('M') or 'MAL' in t: return "Male"
        if t.startswith('F') or t.endswith('F') or 'FEM' in t: return "Female"
        
        # 4. Handle very messy OCR (e.g. 'hl', 'H1', 'i1', 'li')
        # If it starts with h, l, i, etc it might be M/F? No, let's keep it safe.
        # But commonly 'M' is read as 'H' or 'N'.
        if any(k in t for k in ['PU', 'PUR', 'NAT', 'NAL']): return "Male"
        if any(k in t for k in ['MAH', 'STRI', 'W ']): return "Female"
        
        return ""

    @staticmethod
    def map_relation_type(text):
        if not text:
            return ""
        t = str(text).strip().upper()
        
        if any(k in t for k in ['HUSBAND', 'H/O', 'H.O', 'H ']): return 'H'
        if any(k in t for k in ['FATHER', 'S/O', 'S.O', 'F ']): return 'F'
        if any(k in t for k in ['MOTHER', 'D/O', 'D.O', 'M ']): return 'M'
        if any(k in t for k in ['OTHER', 'W/O', 'W.O', 'O ']): return 'O'
        
        # Pass through abbreviations
        if t in ['H', 'F', 'M', 'O']: return t
        
        return ""

    @staticmethod
    def sanitize_name(text, is_relative=False):
        """
        Clean and format English names for voter records.
        """
        if not text:
            return ""
        
        # 1. Basic OCR cleanup
        text = text.replace('||', '').replace('|', '').replace(':', '')
        
        # 2. Remove common English labels and OCR artifacts (like Photo indicators)
        labels_pattern = r'^(?:Father|Husband|Mother|Relative|Other|Name|Nam|Nav|Nam|Phoate|Photo|Pnote|Pinoie|Fnote|Cnoto|Polo|Pnale|Not|Nat)\'?\s*s?\s*(?:Name|Nav|Nam|Not|Available)?\s*(?:is)?\s*[:\-\.\|\\/ ]*'
        text = re.sub(labels_pattern, '', text, flags=re.IGNORECASE).strip()
        
        # 2.1 Specific junk removals for common bleeding text
        junk_patterns = [
            r'\b(?:Phoate|Phote|Pnote|Pinoie|Fnote|Cnoto|Polo|Pnale)\s*(?:Nat|Not)?\b',
            r'\bPhoto\s*(?:Available|Not)?\b'
        ]
        for jp in junk_patterns:
             text = re.sub(jp, '', text, flags=re.IGNORECASE).strip()
        
        # 3. Strictly keep only English letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # 4. Normalize whitespace and capitalize
        text = ' '.join(word.capitalize() for word in text.split()).strip()
        
        # 5. Remove single character noise at start/end
        text = re.sub(r'^[a-z]\s+', '', text, flags=re.IGNORECASE).strip()
        
        return text

    @staticmethod
    def clean_booth_info(text):
        """
        Clean and format booth center/address information (English only).
        """
        if not text:
            return ""
        
        # 1. Remove Watermark Noise
        watermark_patterns = [
            r'STATE\s+ELECTION\s+COMMISSION',
            r'ELECTION\s+COMMISSION\s+OF\s+INDIA',
            r'ELECTION\s+COMMISSION',
            r'STATE\s+ELECTION',
            r'COMMISSION',
            r'ELECTION',
            r'INDIA'
        ]
        active_text = text
        for p in watermark_patterns:
            active_text = re.sub(p, '', active_text, flags=re.IGNORECASE).strip()
        
        # 2. Remove common labels
        labels = [
            r'(?:Polling Station Name|Station Name|Polling Station Address|Address)\s*[:\-]*',
            r'^(?:Center|Address|Station|Name)\b\s*[:\-]*'
        ]
        for lp in labels:
            active_text = re.sub(lp, '', active_text, flags=re.IGNORECASE).strip()

        # 3. Standardize whitespace
        result = ' '.join(active_text.split()).strip()
        
        # 4. Final misread fixes for schools
        result = result.replace('2.2', 'Z.P.').replace('9.P', 'Z.P.').replace('800', 'Bhor')
        
        return result
