from sentence_transformers import util
from .config import EDU_LEVELS, WEIGHT_ELIGIB, WEIGHT_SKILLS, WEIGHT_CONTEXT
from .document_parser import extract_skills_from_text
from .report_generator import get_category_for_skill

def score_eligibility(jd, resume):
    """Compare Education and Experience fields strictly."""
    score = 0.0
    feedback = []
    
    # 1. Experience Check
    req_exp = jd.get('experience_required', {}).get('years', 0)
    cand_exp = resume.get('experience', {}).get('years_detected', 0)
    
    if req_exp == 0 or cand_exp >= req_exp:
        score += 0.5
    else:
        score += 0.1
        feedback.append(f"Short on experience: Has {cand_exp} yrs, needs {req_exp} yrs.")
        
    # 2. Education Check
    req_degrees = jd.get('eligibility', {}).get('degrees_required', [])
    cand_degree = resume.get('education', {}).get('highest_degree', "NOT DETECTED")
    
    cand_norm = cand_degree.replace('.', '').strip()
    cand_level = EDU_LEVELS.get(cand_norm, 0)
    
    req_level = 0
    if req_degrees:
        valid_req_levels = [EDU_LEVELS.get(d.replace('.', '').strip(), 0) for d in req_degrees]
        valid_req_levels = [lvl for lvl in valid_req_levels if lvl > 0]
        if valid_req_levels:
            req_level = min(valid_req_levels)
        
    if req_level == 0:
        score += 0.5
    else:
        if cand_level >= req_level:
            score += 0.5
        elif cand_level == req_level - 1:
            score += 0.35  # Grace for being one level away
            feedback.append(f"Education level is slightly below target (Has {cand_degree}).")
        else:
            score += 0.0
            feedback.append(f"Missing required education degree (Needs {', '.join(req_degrees)}).")
            
    return score, feedback, cand_degree


def score_skills(jd, resume):
    """Compare specific technical skills."""
    req_tech_dict = jd.get('required_skills', {}).get('technical', {})
    req_skills = set()
    for cat, skills in req_tech_dict.items():
        req_skills.update(skills)
        
    # If JD didn't explicitly separate required skills, extract from raw text
    if not req_skills:
        req_text = str(jd.get('responsibilities', {}).get('raw_text', '')) + " " + str(jd.get('company_info', {}).get('raw_text', ''))
        fallback_tech, _ = extract_skills_from_text(req_text)
        for cat, skills in fallback_tech.items():
            req_skills.update(skills)
            req_tech_dict[cat] = skills
        jd.setdefault('required_skills', {})['technical'] = req_tech_dict
    
    cand_tech_dict = resume.get('skills', {}).get('technical', {})
    cand_skills = set()
    for cat, skills in cand_tech_dict.items():
        cand_skills.update(skills)
    
    if not req_skills:
        return 1.0, [], [], []

    matched = req_skills.intersection(cand_skills)
    missing = req_skills.difference(cand_skills)
    
    # --- Intelligent Normalization ---
    # If a JD has 60 skills, matching 15 is actually amazing. 
    # We cap the scoring denominator at 15 to prevent "Skill Bloat" punishment.
    denom = max(5, min(len(req_skills), 15)) 
    score = len(matched) / denom
    score = min(1.0, score) # Cap at 100%
    
    return score, list(matched), list(missing), req_skills


def score_semantic_context(model, jd, resume):
    """
    Compare the semantic meaning of the candidate's Projects/Experience
    against the JD's Responsibilities/Company Info.
    """
    jd_context = jd.get('responsibilities', {}).get('raw_text') or jd.get('company_info', {}).get('raw_text') or ""
    
    cand_context = ""
    cand_context += resume.get('experience', {}).get('raw_text') or ""
    cand_context += " " + (resume.get('projects') or "")
    
    jd_context = str(jd_context)
    cand_context = str(cand_context)
    
    if not jd_context.strip() or not cand_context.strip():
        return 0.0
        
    jd_emb = model.encode(jd_context, convert_to_tensor=True, show_progress_bar=False)
    cand_emb = model.encode(cand_context, convert_to_tensor=True, show_progress_bar=False)
    
    sim = util.cos_sim(jd_emb, cand_emb).item()
    return max(0.0, min(1.0, sim))

def run_what_if_simulation(model, jd, resume, missing_skills, matched_skills, req_skills, final_score, elig_score, context_score):
    top_improvements = []
    if not missing_skills:
        return top_improvements

    jd_context = jd.get('responsibilities', {}).get('raw_text') or jd.get('company_info', {}).get('raw_text') or ""
    jd_context = str(jd_context)
    jd_emb = model.encode(jd_context, convert_to_tensor=True, show_progress_bar=False) if jd_context.strip() else None
    
    cand_base_context = (resume.get('experience', {}).get('raw_text') or "") + " " + (resume.get('projects') or "")
    req_tech_dict = jd.get('required_skills', {}).get('technical', {})
    
    for skill in missing_skills:
        augmented_context = cand_base_context + f" Proficient in {skill}. Hands-on experience with {skill}."
        
        if jd_emb is not None:
            cand_emb = model.encode(augmented_context, convert_to_tensor=True, show_progress_bar=False)
            new_context_score = max(0.0, min(1.0, util.cos_sim(jd_emb, cand_emb).item()))
        else:
            new_context_score = context_score
        
        new_count = len(matched_skills) + 1
        denom = max(5, min(len(req_skills), 15))
        new_skill_score = min(1.0, new_count / denom)
        new_final = (elig_score * WEIGHT_ELIGIB) + (new_skill_score * WEIGHT_SKILLS) + (new_context_score * WEIGHT_CONTEXT)
        impact = new_final - final_score
        
        skill_cat = get_category_for_skill(skill, req_tech_dict)
                
        top_improvements.append({
            "skill": skill,
            "impact_pct": impact,
            "new_pct": new_final,
            "category": skill_cat
        })
    
    top_improvements.sort(key=lambda x: x['impact_pct'], reverse=True)
    return top_improvements[:5]
