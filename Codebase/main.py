import os
import json
import glob
from datetime import datetime

from Module.config import (
    INPUT_DIR, JD_DIR, RESUME_DIR, OUTPUT_DIR, RESUME_JSON_OUT_DIR, JD_JSON_OUT_DIR, 
    PROCESSED_JD_PATH, RESULTS_PATH, 
    WEIGHT_SKILLS, WEIGHT_CONTEXT, WEIGHT_ELIGIB,
    RESUME_SCHEMA, JD_SCHEMA
)
from Module.logger import logger
from Module.utils import ModelLoader
from Module.document_parser import (
    parse_jd, parse_resume, build_canonical_embeddings
)
from Module.matcher_engine import (
    score_eligibility, score_skills, score_semantic_context, run_what_if_simulation
)
from Module.report_generator import (
    generate_smart_feedback, generate_full_report
)

def run_parser():
    logger.info("--- Phase 1: Document Parsing ---")
    model = ModelLoader.get_model()

    logger.info("Building section embeddings for Resume schema...")
    resume_embeddings = build_canonical_embeddings(RESUME_SCHEMA, model)

    logger.info("Building section embeddings for JD schema...")
    jd_embeddings = build_canonical_embeddings(JD_SCHEMA, model)

    # Find the JD file (either txt or pdf) in JD_DIR
    jd_files = glob.glob(os.path.join(JD_DIR, "*.txt")) + glob.glob(os.path.join(JD_DIR, "*.pdf"))
    if not jd_files:
        logger.error(f"No Job Description found in {JD_DIR}")
        return False
        
    jd_file_path = jd_files[0]  # Just take the first one found
    logger.info(f"Parsing JD: {os.path.basename(jd_file_path)}")
    jd_data = parse_jd(jd_file_path, model, jd_embeddings, JD_SCHEMA)
    with open(PROCESSED_JD_PATH, 'w', encoding='utf-8') as f:
        json.dump(jd_data, f, indent=2, ensure_ascii=False)
    logger.info(f"[OK] JD saved -> {PROCESSED_JD_PATH}")

    resume_files = (
        glob.glob(os.path.join(RESUME_DIR, "*.pdf")) +
        glob.glob(os.path.join(RESUME_DIR, "*.txt"))
    )
    
    if not resume_files:
        logger.error(f"No resumes found in {RESUME_DIR}")
        return False

    logger.info(f"Found {len(resume_files)} resume(s). Parsing...")
    for file_path in resume_files:
        logger.info(f"Parsing resume: {os.path.basename(file_path)}")
        profile = parse_resume(file_path, model, resume_embeddings, RESUME_SCHEMA)
        
        out_name = os.path.basename(file_path).replace('.pdf', '').replace('.txt', '') + '.json'
        out_path = os.path.join(RESUME_JSON_OUT_DIR, out_name)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        logger.info(f"[OK] Saved -> {out_path}")

    logger.info(f"Done. All JD JSONs saved in: {JD_JSON_OUT_DIR}/")
    logger.info(f"Done. All Resume JSONs saved in: {RESUME_JSON_OUT_DIR}/")
    return True

def run_matcher():
    logger.info("--- Phase 2: Semantic Matching ---")
    if not os.path.exists(PROCESSED_JD_PATH):
        logger.error(f"JD Profile not found at {PROCESSED_JD_PATH}. Ensure parsing was successful.")
        return

    try:
        with open(PROCESSED_JD_PATH, 'r', encoding='utf-8') as f:
            jd_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to load JD JSON: {e}")
        return

    resume_files = glob.glob(os.path.join(RESUME_JSON_OUT_DIR, '*.json'))

    if not resume_files:
        logger.error(f"No processed resumes found in {RESUME_JSON_OUT_DIR}.")
        return

    model = ModelLoader.get_model()
    
    logger.info(f"Analyzing {len(resume_files)} Candidate Profiles against the Job Description...")
    logger.info(f"Target Role: {jd_data.get('role_summary', {}).get('title', 'Not Found')}")
    logger.info(f"Company: {jd_data.get('company_info', {}).get('name', 'Not Found')}")
    print("-" * 60)

    results = []

    for r_file in resume_files:
        try:
            with open(r_file, 'r', encoding='utf-8') as f:
                resume = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping corrupted JSON {r_file}: {e}")
            continue
            
        cand_name = resume.get('personal_info', {}).get('name') or os.path.basename(r_file).replace('.json', '')
        
        # 1. Eligibility (Strict matching)
        elig_score, elig_feedback, cand_degree = score_eligibility(jd_data, resume)
        
        # 2. Skills (Direct intersection)
        skill_score, matched_skills, missing_skills, req_skills = score_skills(jd_data, resume)
        
        # 3. Context (Semantic understanding of Projects/Exp vs JD)
        context_score = score_semantic_context(model, jd_data, resume)
        
        # Compute Base Hybrid Score
        final_score = (elig_score * WEIGHT_ELIGIB) + (skill_score * WEIGHT_SKILLS) + (context_score * WEIGHT_CONTEXT)
        
        # 4. What-If Simulator (The X-Factor)
        top_improvements = run_what_if_simulation(
            model, jd_data, resume, missing_skills, matched_skills, req_skills, final_score, elig_score, context_score
        )
        
        # Generate Feedback
        cand_tech_dict = resume.get('skills', {}).get('technical', {})
        req_tech_dict = jd_data.get('required_skills', {}).get('technical', {})
        feedback = generate_smart_feedback(
            final_score, skill_score, context_score, elig_score,
            cand_tech_dict, req_tech_dict, matched_skills, 
            missing_skills, elig_feedback, top_improvements
        )
        
        results.append({
            "name": cand_name,
            "final_pct": round(final_score * 100, 1),
            "elig_pct": round(elig_score * 100, 1),
            "skill_pct": round(skill_score * 100, 1),
            "context_pct": round(context_score * 100, 1),
            "cand_degree": cand_degree,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "top_improvements": top_improvements,
            "feedback": feedback
        })

    results.sort(key=lambda x: x['final_pct'], reverse=True)

    # Output to Console and TXT
    report_text = generate_full_report(results)
    
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)
        
    for line in report_text.split('\n'):
        logger.info(line)
        
    logger.info(f"[OK] Detailed report saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    if run_parser():
        run_matcher()
