from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import re

class TranslitHelper:
    # Use simple cache for transliteration to speed up performance
    _translit_cache = {}

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
    def clean_booth_info(text):
        """
        Cleans booth info text from OCR artifacts.
        Corrects '2' or '9' or other numeric misreads for 'Z.P.'
        """
        if not text:
            return ""
        
        # 1. Standardize whitespace
        text = ' '.join(text.split()).strip()
        
        # 2. Specific English/Marathi misread corrections (Common OCR errors)
        # Fix '800' -> 'Bhor'
        text = re.sub(r'\b800\b', 'Bhor', text)
        # Fix '28' -> 'Zilla' if near school keywords
        text = re.sub(r'\b28\b(?=\s*(?:जि|प|प्रा|shala|school|pri|pra|केंद्र))', 'Zilla', text, flags=re.IGNORECASE)
        # Fix 'He' -> 'Kendra' (Common OCR misread for केंद्र or के.)
        text = re.sub(r'\bHe[.]?\b', 'Kendra', text)
        # Fix 'पब' -> 'पु' (if it slipped into here)
        text = text.replace('पब', 'पु')
        
        # 3. Keywords indicating Zila Parishad schools
        # English and Marathi variants
        school_keywords = ['pri', 'pra', 'school', 'shala', 'primary', 'प्रा', 'प्राथ', 'शाळा', 'केंद्र']
        
        # 4. Handle suspicious starts followed by school keywords
        # User reported 2 or 9 replacing Z.P.
        # Pattern: Starting with 2, 9, 8, 0, Z, etc. followed by a school keyword
        target_kws_pattern = '|'.join([re.escape(k) for k in school_keywords])
        
        # Case 1: Suspicious leading chars + keyword
        # Corrects: "2 pri school" -> "Z.P. pri school", "9 pra. school" -> "Z.P. pra. school"
        sus_pattern = r'^([2980Zz\.\s,]+)\s+(?=' + target_kws_pattern + ')'
        if re.search(sus_pattern, text, re.IGNORECASE):
            text = re.sub(sus_pattern, 'Z.P. ', text, count=1, flags=re.IGNORECASE)
        
        # 5. Case 2: Direct numeric fragments that look like Z.P.
        # "2.2" -> "Z.P.", "2,2" -> "Z.P.", "2 P" -> "Z.P."
        text = re.sub(r'\b2[,. ]+2\b', 'Z.P.', text)
        text = re.sub(r'\b2[,. ]+P\b', 'Z.P.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bZ[,. ]+2\b', 'Z.P.', text, flags=re.IGNORECASE)
        text = re.sub(r'\b9[,. ]+P\b', 'Z.P.', text, flags=re.IGNORECASE)
        text = re.sub(r'\b9[,. ]+2\b', 'Z.P.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bZ[,. ]+P\b', 'Z.P.', text, flags=re.IGNORECASE)
        
        # 6. Case 3: Specific Marathi Ji.Pa. (जि.प.) misreads
        # If it starts with 2 or 9 and then Marathi school keywords
        mar_kws = ['प्रा', 'प्राथ', 'शाळा', 'केंद्र']
        mar_target = '|'.join(mar_kws)
        mar_sus = r'^([2980])\s+(?=' + mar_target + ')'
        if re.search(mar_sus, text):
            text = re.sub(mar_sus, 'जि.प. ', text, count=1)

        return text.strip()

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
        text = re.sub(r'[होखोलोसोशो]ख', 'शेख', text)
        
        # 3. General cleaning for names (Remove colons and leading garbage)
        # remove ":" automatically if the grid misalligned
        text = re.sub(r'^[:\- .]*', '', text)
        text = re.sub(r'[:]', '', text) # Global colon removal
        
        return text.strip()

    @staticmethod
    def transliterate_marathi_to_english(text):
        """
        Phonetic transliteration with modern naming conventions (Schwa deletion).
        Fixes 'Patila' -> 'Patil', 'Dilipa' -> 'Dilip', etc.
        """
        if not text:
            return ""
        
        # Performance Cache Check
        if text in TranslitHelper._translit_cache:
            return TranslitHelper._translit_cache[text]
        
        original_text = text
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
                
            # Specific name correction: 'JNaneshVar' -> 'Dnyneshwar' as requested
            if 'jnaneshvar' in t_word_lower or 'jnaneshwar' in t_word_lower:
                t_word = 'Dnyneshwar'
            elif 'jnyaneshwar' in t_word_lower:
                t_word = 'Dnyneshwar'
                
            # Capitalize
            t_word = t_word.title()
            
            # Clean non-ASCII (except spaces and brackets if needed)
            # Clean non-ASCII (except spaces and brackets if needed)
            t_word = re.sub(r'[^a-zA-Z0-9\(\)]', '', t_word)
            
            transliterated_words.append(t_word)
            
        result = " ".join(transliterated_words).strip()
        
        # Save to cache before returning
        TranslitHelper._translit_cache[original_text] = result
        return result

    @staticmethod
    def map_gender(marathi_gender):
        if not marathi_gender:
            return ""
        m_gender = str(marathi_gender).strip().upper()
        
        # Male keywords: पु, पब (OCR error), पुरुष (Hindi/Marathi)
        if any(k in m_gender for k in ['पब', 'पम', 'पर', 'पु', 'पुरुष', 'PURUSH', 'MALE', 'PURS']):
             return "Male"
        
        # Female keywords: स्री, स्त्री, सण (OCR error), महिला (Hindi), सत, सद
        if any(k in m_gender for k in ['सण', 'सद', 'सत', 'स्री', 'स्त्री', 'महिला', 'MAHILA', 'FEMALE', 'STRI']):
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
