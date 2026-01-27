from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import re

class TranslitHelper:
    # Specific OCR correction map for common Marathi name/surname confusions
    MARATHI_CORRECTIONS = {
        'ठोळके': 'शेळके',
        'क्रषिकेश': 'ऋषिकेश',
        'आशिविनी': 'अश्विनी',
        'आशिवनी': 'अश्विनी',
        'काशोनाथ': 'काशिनाथ',
        'भास्कर': 'भास्कर',
        'यठावंत': 'यशवंत',
        'सुरेठठा': 'सुरेश',
        'सुरेठठ': 'सुरेश',
        'होख': 'शेख',
        'सोख': 'शेख',
        'लोख': 'शेख',
        'हडोख': 'शेख',
        'चोख': 'शेख',
        'हडठोख': 'शेख',
        'घोष': 'शेख',
        'शोख': 'शेख',
        'लाख': 'शेख',
        'शख': 'शेख',
        'होष': 'शेख',
        'होप': 'शेख',
        'होाख': 'शेख',
        'माटेवाची': 'माटेवाडी',
        'नाटेवाची': 'माटेवाडी',
        'माटेवाचि': 'माटेवाडी',
        'वाची': 'वाडी',
        'वाचि': 'वाडी',
    }

    # Pre-compiled regex for speed
    RE_SHAIKH = re.compile(r'\b(हो|सो|लो|लोग|चो|शो|हडो|हडठो|होड|चोर|हाड|होा|स्रो)[ ]?(ख|प|ष)\b')
    RE_THOLKE = re.compile(r'\bठो(ळके)\b')
    RE_RISHI = re.compile(r'क्रषि')
    RE_ASHVI = re.compile(r'आशिव(ि|इ)?')
    RE_WADI = re.compile(r'([ा-ो])वा(ची|चि)\b')

    @staticmethod
    def correct_marathi_ocr(text):
        """
        Corrects specific Marathi OCR misidentifications.
        Only targets words that are highly likely to be OCR errors.
        """
        if not text:
            return ""
        
        # 1. Targeted word replacements for known OCR errors (Fast literal replace)
        for wrong, right in TranslitHelper.MARATHI_CORRECTIONS.items():
            if wrong in text:
                text = text.replace(wrong, right)
        
        # 2. Pattern-based correction for common OCR confusions
        
        # Fix 'वाची/वाचि' -> 'वाडी' (Place name suffixes)
        text = TranslitHelper.RE_WADI.sub(r'\1वाडी', text)
        
        # Fix 'क्रषि' -> 'ऋषि'
        text = TranslitHelper.RE_RISHI.sub('ऋषि', text)
        
        # Fix 'आशिव' -> 'अश्व'
        text = TranslitHelper.RE_ASHVI.sub('अश्व', text)
        
        # Fix 'ठो' -> 'शे' before 'ळके'
        text = TranslitHelper.RE_THOLKE.sub(r'शे\1', text)
        
        # General fix for Shaikh (शेख) misidentifications
        # User reported 'hokh' specifically.
        # We handle variations with optional spaces between characters.
        text = TranslitHelper.RE_SHAIKH.sub('शेख', text)
        
        return text

    @staticmethod
    def transliterate_marathi_to_english(text):
        """
        Phonetic transliteration with modern naming conventions (Schwa deletion).
        Fixes 'Patila' -> 'Patil', 'Dilipa' -> 'Dilip', etc.
        """
        if not text:
            return ""
        
        # Correct OCR errors in Marathi source first
        text = TranslitHelper.correct_marathi_ocr(text)
        
        # Pre-process specific terms
        text = text.replace('(एच)', '(H)')
        
        # Use ITRANS which is very reliable for phonetics
        words = text.split()
        transliterated_words = []
        
        for word in words:
            # Handle specific keywords manually if needed
            if word == '(H)':
                transliterated_words.append('(H)')
                continue

            # Transliterate to ITRANS
            t_word = transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
            
            # MODERN NAMING CONVENTION (Schwa Deletion Logic):
            # 1. If word ends in 'aa' (derived from 'ा'), it should be a single 'a' in modern English names
            if t_word.endswith('aa'):
                t_word = t_word[:-2] + 'a'
            
            # 2. If word ends in a single 'a' (derived from inherent schwa), it should be REMOVED for modern English names
            elif t_word.endswith('a') and len(t_word) > 2:
                # Check if it should actually remain (e.g., Rama, Krishna - though in MH names usually Ram, Krishn)
                # In MH, we almost always drop the final 'a'
                t_word = t_word[:-1]
                
            # Post-process common ITRANS characters to standard English
            t_word = t_word.replace('shhe', 'she').replace('Shhe', 'She')
            t_word = t_word.replace('shh', 'sh').replace('Shh', 'Sh')
            t_word = t_word.replace('ii', 'i').replace('uu', 'u')
            t_word = t_word.replace('aa', 'a') 
            t_word = t_word.replace('.n', 'n') # Use 'n' for anusvara
            t_word = t_word.replace('.m', 'n') # ITRANS often uses .m for Anusvara, map to 'n'
            t_word = t_word.replace('M', 'n')  # Some contexts use capital M for Anusvara -> map to 'n' for simpler reading
            
            # Specific surname/name transliteration overrides
            # 'shekh' (शेख), 'shokh' (शोख), 'hokh' (होख) -> 'Shaikh'
            t_word_lower = t_word.lower()
            if t_word_lower in ['shekh', 'shokh', 'hokh', 'sokh', 'lokh', 'hokha']:
                t_word = 'Shaikh'
            elif t_word_lower == 'patila':
                t_word = 'Patil'
            elif t_word.lower() == 'shinde':
                t_word = 'Shinde' # ITRANS might make it different
            elif t_word.lower() == 'deshamukha':
                t_word = 'Deshmukh'
            elif t_word.lower() == 'pavar':
                t_word = 'Pawar'
                
            # Fix 'Ri' for vocalic R (ऋ)
            t_word = t_word.replace('RRi', 'Ri')
            
            # Capitalize
            t_word = t_word.title()
            
            # Clean non-ASCII (except spaces and brackets if needed)
            # Keeping brackets for (H)
            t_word = re.sub(r'[^a-zA-Z0-9\(\)]', '', t_word)
            
            transliterated_words.append(t_word)
            
        return " ".join(transliterated_words).strip()

    @staticmethod
    def map_gender(marathi_gender):
        if not marathi_gender:
            return ""
        m_gender = str(marathi_gender).strip().upper()
        
        # Marathi & Hindi Gender Keywords
        # Male keywords: पु, पुरुष (Hindi/Marathi)
        if any(k in m_gender for k in ['पम', 'पर', 'पु', 'पुरुष', 'PURUSH', 'MALE', 'PURS']):
             return "Male"
        
        # Female keywords: स्री, स्त्री, महिला (Hindi), सत, सद
        if any(k in m_gender for k in ['सद', 'सत', 'स्री', 'स्त्री', 'महिला', 'MAHILA', 'FEMALE', 'STRI']):
             return "Female"
        
        return ""

    @staticmethod
    def map_relation_type(marathi_type_code):
        """
        Maps Marathi relation terms to standard codes:
        H (Husband), F (Father), M (Mother), O (Other)
        """
        if not marathi_type_code:
            return ""
            
        code = str(marathi_type_code).strip()
        
        # 1. HUSBAND: 'पती', 'पतीचे', 'पति' (Hindi) -> 'H'
        for k in ['पतीचे नाव', 'पतीचे', 'पती', 'पति', 'पतीचेनाव', 'पति का नाम', 'पति']:
            if k in code:
                return 'H'

        # 2. FATHER: 'वडिलांचे', 'वडील', 'वलडल', 'पिता' (Hindi) -> 'F'
        for k in ['वडिलांचे नाव', 'वडिलांचे', 'वडील', 'वलडल', 'पिता', 'पिता का नाम', 'वडिलांचेनाव']:
            if k in code:
                return 'F'
             
        # 3. MOTHER: 'आई', 'माता' (Hindi) -> 'M'
        for k in ['आईचे नाव', 'आई', 'आईचे', 'आईचेनाव', 'माता', 'माता का नाम']:
            if k in code:
                return 'M'
             
        # 4. OTHER / SELF: 'इतर', 'अन्य' (Hindi) -> 'O'
        for k in ['इतर', 'स्वतः', 'इतरनाव', 'अन्य', 'अन्य का']:
            if k in code:
                return 'O'

        # Fallback logic if no clear keyword found
        if 'प' in code and len(code) < 5: return 'H'
        if 'व' in code and len(code) < 5: return 'F'
        
        return 'O'

    @staticmethod
    def transliterate_relation_type(marathi_type):
        """
        Transliterates the relation type to English using codes or keywords.
        """
        if not marathi_type:
            return ""
            
        mapping = {
            'H': 'Husband',
            'F': 'Father',
            'M': 'Mother',
            'O': 'Other',
            'पतीचे नाव': 'Husband',
            'पती': 'Husband',
            'पति': 'Husband',
            'पति का नाम': 'Husband',
            'पतीचे': 'Husband',
            'वडिलांचे नाव': 'Father',
            'वडील': 'Father',
            'पिता': 'Father',
            'पिता का नाम': 'Father',
            'वलडल': 'Father',
            'वडिलांचे': 'Father',
            'आईचे नाव': 'Mother',
            'आई': 'Mother',
            'माता': 'Mother',
            'माता का नाम': 'Mother',
            'इतर': 'Other',
            'अन्य': 'Other'
        }
        
        # Check standard codes first
        if marathi_type in mapping:
            return mapping[marathi_type]

        # Fallback to search if it's still raw text
        for k, v in mapping.items():
            if k in marathi_type:
                return v
                
        return marathi_type

    @staticmethod
    def smart_correct_name(full_name_marathi, relative_name_marathi):
        """
        AI-lite Contextual Correction Model:
        Cross-references the Person's Name with the separately extracted Relative Name.
        
        Logic:
        1. Parse Full Name: [Surname] [Own Name] [Relative Name]
        2. Parse Relative Name: [Surname] [First Name]
        3. Compare [Relative Name] in Full Name vs [First Name] in Relative Name.
        4. If fuzzy match > 80% (but not perfect), replace the bad OCR version in Full Name 
           with the clean version from Relative Name.
           
        Example:
        Full Name: "Latavade Bhagyashri Surethatha" (Error)
        Relative Name: "Latavade Suresh" (Clean)
        Result: "Latavade Bhagyashri Suresh"
        """
        if not full_name_marathi or not relative_name_marathi:
            return full_name_marathi
            
        try:
            import difflib
            
            # 1. Clean and Split
            f_parts = full_name_marathi.strip().split()
            r_parts = relative_name_marathi.strip().split()
            
            if len(f_parts) < 3 or len(r_parts) < 2:
                return full_name_marathi # Not enough parts to analyze
                
            # Usually: 
            # Full Name = [Surname] [Own] [Husband/Father]
            # Relative Name = [Surname] [Husband/Father]
            
            # Candidate 1: Husband/Father name in Full Name is typically the LAST word
            candidate_bad = f_parts[-1]
            
            # Candidate 2: Husband/Father name in Relative Name is typically the LAST word (Surname + First)
            # BUT sometimes Relative Name is just First Name.
            # Let's assume standard format "Surname Firstname" -> so it's the 2nd word.
            candidate_good = r_parts[-1] # Take last word of relative name
            
            # Calculate Similarity
            similarity = difflib.SequenceMatcher(None, candidate_bad, candidate_good).ratio()
            
            # Threshold: 0.60 (60% match) - Lowered to catch 'Surethatha' (10) vs 'Suresh' (6) -> ~0.625
            if 0.60 < similarity < 1.0:
                print(f"      AI Correction: Replacing '{candidate_bad}' with '{candidate_good}' (Sim: {similarity:.2f})")
                
                # Replace the last word
                f_parts[-1] = candidate_good
                return " ".join(f_parts)
                
            return full_name_marathi
            
        except Exception as e:
            print(f"      AI Helper Error: {e}")
            return full_name_marathi
