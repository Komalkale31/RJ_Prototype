import os
import re
import datetime
import pdfplumber
from sentence_transformers import util
from .config import SIM_THRESHOLD, MAX_HEADER_LEN, ALL_TECH_SKILLS, SOFT_SKILLS

# ── Extraction Helpers ───────────────────────────────────────────────

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', text)
    if match:
        phone = re.sub(r'[^\d+]', '', match.group(0))
        if len(phone) >= 10:
            return phone
    return None

def extract_social_links(text):
    linkedin = re.search(r'(https?://(?:www\.)?linkedin\.com/in/[^\s]+)', text)
    github = re.search(r'(https?://(?:www\.)?github\.com/[^\s]+)', text)
    hf = re.search(r'(https?://(?:www\.)?huggingface\.co/[^\s]+)', text)
    return {
        "linkedin": linkedin.group(0) if linkedin else None,
        "github": github.group(0) if github else None,
        "huggingface": hf.group(0) if hf else None
    }

def extract_skills_from_text(text):
    text_lower = text.lower()
    technical = {}
    soft = []
    
    # Extract technical skills with categorization
    for category, skill in ALL_TECH_SKILLS:
        skill_clean = skill.lower()
        # Case 1: Standard word match (most accurate)
        pattern = r'\b' + re.escape(skill_clean) + r'\b'
        if re.search(pattern, text_lower):
            technical.setdefault(category, []).append(skill)
            continue
            
        # Case 2: Concatenated match (e.g. SQLPython)
        # We check if the skill exists as a substring, but verify it's not a tiny skill inside a bigger word (like 'it' inside 'hit')
        if len(skill_clean) > 3 and skill_clean in text_lower:
             technical.setdefault(category, []).append(skill)
            
    # Sort the lists within the categories
    technical = {k: sorted(v) for k, v in technical.items()}
    
    for skill in SOFT_SKILLS:
        skill_clean = skill.lower()
        if re.search(r'\b' + re.escape(skill_clean) + r'\b', text_lower):
            soft.append(skill)
        elif len(skill_clean) > 5 and skill_clean in text_lower:
            soft.append(skill)
            
    return technical, sorted(set(soft))

def extract_education_info(text):
    text_lower = text.lower()
    degree_map = {
        "phd": "PHD", "doctorate": "DOCTORATE",
        "mtech": "MTECH", "m.tech": "MTECH", "mca": "MCA", "msc": "MSC", "m.sc": "MSC", "me ": "ME", "m.e.": "ME", "mba": "MBA", "master": "MASTER", "masters": "MASTER",
        "btech": "BTECH", "b.tech": "BTECH", "bca": "BCA", "bsc": "BSC", "b.sc": "BSC", "be ": "BE", "b.e.": "BE", "bachelor": "BACHELOR",
        "diploma": "DIPLOMA",
    }
    
    highest_degree = "NOT DETECTED"
    for keyword, mapped_degree in degree_map.items():
        if keyword in text_lower:
            highest_degree = mapped_degree
            break 
            
    return {
        "raw_text": text.strip() if text.strip() else None,
        "highest_degree": highest_degree,
        "certifications": []
    }

def extract_years_experience(text):
    if not text.strip():
        return 0
    text_lower = text.lower()
    if 'intern' in text_lower and 'full time' not in text_lower:
        return 0
    match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)', text_lower)
    if match:
        return int(match.group(1))
    return 0

def extract_jd_metadata(text):
    text_lower = text.lower()
    batch_year = None
    match = re.search(r'(20\d{2})\s*(?:passout|batch|graduates?)', text_lower)
    if match:
        batch_year = match.group(1)
        
    degrees = []
    degree_map = {
        "mtech": "MTECH", "mca": "MCA", "msc": "MSC", "me": "ME", "mba": "MBA",
        "btech": "BTECH", "bca": "BCA", "bsc": "BSC", "be": "BE"
    }
    for kw, mapped in degree_map.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower) or re.search(r'\b' + re.escape(kw[0] + '.' + kw[1:]) + r'\b', text_lower):
            degrees.append(mapped)
            
    min_pct = None
    pct_match = re.search(r'(\d{2})%', text_lower)
    if pct_match:
        min_pct = pct_match.group(0)
        
    return batch_year, list(set(degrees)), min_pct

def extract_ctc(text):
    match = re.search(r'((?:rs\.?|inr|₹)?\s*\d+(?:\.\d+)?\s*(?:lpa|l|k|cr))', text.lower())
    return match.group(1).upper() if match else None

# ── Semantic Processing ──────────────────────────────────────────────

def build_canonical_embeddings(schema, model):
    embeddings = {}
    for section, keywords in schema.items():
        keyword_str = " ".join(keywords)
        emb = model.encode(keyword_str, convert_to_tensor=True, show_progress_bar=False)
        embeddings[section] = emb
    return embeddings

def detect_sections(lines, canonical_embeddings, model):
    detected_blocks = {}
    current_section = None
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        is_header = False
        words = line_clean.split()
        
        if len(words) <= MAX_HEADER_LEN and (line_clean.isupper() or line_clean.istitle() or line_clean.endswith(':')):
            line_emb = model.encode(line_clean, convert_to_tensor=True, show_progress_bar=False)
            best_score = 0
            best_section = None
            
            for section, can_emb in canonical_embeddings.items():
                score = util.cos_sim(line_emb, can_emb).item()
                if score > best_score:
                    best_score = score
                    best_section = section
                    
            if best_score >= SIM_THRESHOLD:
                current_section = best_section
                is_header = True
                if current_section not in detected_blocks:
                    detected_blocks[current_section] = []
                    
        if current_section and not is_header:
            detected_blocks[current_section].append(line_clean)
            
    for sec in detected_blocks:
        detected_blocks[sec] = "\n".join(detected_blocks[sec])
        
    return detected_blocks

# ── Pipeline Wrappers ────────────────────────────────────────────────

def extract_text_from_image_pdf(file_path):
    text = ""
    try:
        import fitz
        import easyocr
        import numpy as np
        
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            result = reader.readtext(img, detail=0)
            text += " ".join(result) + "\n"
    except ImportError:
        print(f"    [!] easyocr or pymupdf not installed. Skipping OCR for {os.path.basename(file_path)}")
    except Exception as e:
        print(f"    [!] OCR Error on {file_path}: {e}")
    return text

def read_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        
    if len(text.strip()) < 50:
        print(f"    [!] Scanned PDF detected. Attempting OCR on {os.path.basename(file_path)}...")
        text = extract_text_from_image_pdf(file_path)
        
    return text

def parse_resume(file_path, model, embeddings, schema):
    if file_path.lower().endswith('.pdf'):
        text = read_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

    lines = text.split('\n')
    blocks = detect_sections(lines, embeddings, model)
    
    pre_header_text = "\n".join(lines[:30])
    email = extract_email(pre_header_text)
    phone = extract_phone(pre_header_text)
    socials = extract_social_links(pre_header_text)
    name = lines[0].strip() if lines else "Unknown"

    tech_skills, soft_skills = extract_skills_from_text(text)
    edu_data = extract_education_info(text)
    years_exp = extract_years_experience(blocks.get("experience", ""))
    
    profile = {
        "meta": {
            "source_file": os.path.basename(file_path),
            "processed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sections_detected": list(blocks.keys())
        },
        "personal_info": {
            "name": name,
            "email": email,
            "phone": phone,
            **socials
        },
        "objective_summary": blocks.get("objective_summary"),
        "skills": {
            "technical": tech_skills,
            "soft": soft_skills,
            "raw_text": blocks.get("skills")
        },
        "experience": {
            "raw_text": blocks.get("experience"),
            "years_detected": years_exp
        },
        "projects": blocks.get("projects"),
        "specialized_it": blocks.get("specialized_it"),
        "education": edu_data,
        "certifications": blocks.get("certifications"),
        "achievements": blocks.get("achievements"),
        "research": blocks.get("research"),
        "activities": blocks.get("activities"),
        "additional_sections": blocks.get("additional"),
        "references": blocks.get("references")
    }
    return profile

def parse_jd(file_path, model, embeddings, schema):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    lines = text.split('\n')
    blocks = detect_sections(lines, embeddings, model)
    
    company_name = lines[0].strip() if lines else "Unknown"
    company_loc = []
    if "pune" in text.lower(): company_loc.append("Pune")
    
    batch, degrees, min_pct = extract_jd_metadata(blocks.get("eligibility", "") or text)
    req_tech, req_soft = extract_skills_from_text(blocks.get("required_skills", ""))
    
    ctc = extract_ctc(blocks.get("benefits", ""))
    
    profile = {
        "meta": {
            "source_file": os.path.basename(file_path),
            "processed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sections_detected": list(blocks.keys())
        },
        "company_info": {
            "name": company_name,
            "location": company_loc,
            "raw_text": blocks.get("company_info")
        },
        "role_summary": {
            "title": blocks.get("role_summary", "").split('\n')[0] if blocks.get("role_summary") else "Unknown",
            "raw_text": blocks.get("role_summary")
        },
        "responsibilities": {
            "key_points": [],
            "raw_text": blocks.get("responsibilities")
        },
        "required_skills": {
            "technical": req_tech,
            "raw_text": blocks.get("required_skills")
        },
        "good_to_have": {
            "technical": extract_skills_from_text(blocks.get("good_to_have", ""))[0],
            "raw_text": blocks.get("good_to_have")
        },
        "eligibility": {
            "batch_year": batch,
            "degrees_required": degrees,
            "min_percentage": min_pct,
            "raw_text": blocks.get("eligibility")
        },
        "experience_required": {
            "years": extract_years_experience(blocks.get("experience_required", "")),
            "raw_text": blocks.get("experience_required")
        },
        "benefits": {
            "ctc": ctc,
            "raw_text": blocks.get("benefits")
        },
        "domains": [],
        "process": blocks.get("process"),
        "expectations": blocks.get("expectations"),
        "work_environment": blocks.get("work_environment"),
        "legal_closing": blocks.get("legal_closing")
    }
    return profile
