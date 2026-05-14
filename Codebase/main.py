import os
import json
import glob
from datetime import datetime
from sentence_transformers import util

from Module.config import (
    INPUT_DIR, JD_DIR, RESUME_DIR, OUTPUT_DIR, RESUME_JSON_OUT_DIR, JD_JSON_OUT_DIR, 
    PROCESSED_JD_PATH, RESULTS_PATH, 
    MODEL_NAME, WEIGHT_SKILLS, WEIGHT_CONTEXT, WEIGHT_ELIGIB,
    RESUME_SCHEMA, JD_SCHEMA
)
from Module.utils import ModelLoader
from Module.document_parser import (
    parse_jd, parse_resume, build_canonical_embeddings
)
from Module.matcher_engine import (
    score_eligibility, score_skills, score_semantic_context
)
from Module.report_generator import (
    generate_smart_feedback, get_category_for_skill
)

def run_parser():
    print(f"\n--- Phase 1: Document Parsing ---")
    model = ModelLoader.get_model()

    print("  Building section embeddings for Resume schema...")
    resume_embeddings = build_canonical_embeddings(RESUME_SCHEMA, model)

    print("  Building section embeddings for JD schema...")
    jd_embeddings = build_canonical_embeddings(JD_SCHEMA, model)

    # Find the JD file (either txt or pdf) in JD_DIR
    jd_files = glob.glob(os.path.join(JD_DIR, "*.txt")) + glob.glob(os.path.join(JD_DIR, "*.pdf"))
    if not jd_files:
        print(f"  [!] No Job Description found in {JD_DIR}")
        return False
        
    jd_file_path = jd_files[0]  # Just take the first one found
    print(f"    Parsing JD: {os.path.basename(jd_file_path)}")
    jd_data = parse_jd(jd_file_path, model, jd_embeddings, JD_SCHEMA)
    with open(PROCESSED_JD_PATH, 'w', encoding='utf-8') as f:
        json.dump(jd_data, f, indent=2, ensure_ascii=False)
    print(f"    [OK] JD saved -> {PROCESSED_JD_PATH}")

    resume_files = (
        glob.glob(os.path.join(RESUME_DIR, "*.pdf")) +
        glob.glob(os.path.join(RESUME_DIR, "*.txt"))
    )
    
    if not resume_files:
        print(f"  [!] No resumes found in {RESUME_DIR}")
        return False

    print(f"\n  Found {len(resume_files)} resume(s). Parsing...")
    for file_path in resume_files:
        print(f"    Parsing resume: {os.path.basename(file_path)}")
        profile = parse_resume(file_path, model, resume_embeddings, RESUME_SCHEMA)
        
        out_name = os.path.basename(file_path).replace('.pdf', '').replace('.txt', '') + '.json'
        out_path = os.path.join(RESUME_JSON_OUT_DIR, out_name)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        print(f"    [OK] Saved -> {out_path}")

    print(f"\n  Done. All JD JSONs saved in: {JD_JSON_OUT_DIR}/")
    print(f"  Done. All Resume JSONs saved in: {RESUME_JSON_OUT_DIR}/")
    return True

def run_matcher():
    print(f"\n--- Phase 2: Semantic Matching ---")
    if not os.path.exists(PROCESSED_JD_PATH):
        print(f"[!] JD Profile not found at {PROCESSED_JD_PATH}. Ensure parsing was successful.")
        return

    with open(PROCESSED_JD_PATH, 'r', encoding='utf-8') as f:
        jd_data = json.load(f)

    resume_files = glob.glob(os.path.join(RESUME_JSON_OUT_DIR, '*.json'))

    if not resume_files:
        print(f"[!] No processed resumes found in {RESUME_JSON_OUT_DIR}.")
        return

    model = ModelLoader.get_model()
    
    print(f"\n  Analyzing {len(resume_files)} Candidate Profiles against the Job Description...")
    print(f"  Target Role: {jd_data.get('role_summary', {}).get('title', 'Not Found')}")
    print(f"  Company: {jd_data.get('company_info', {}).get('name', 'Not Found')}")
    print("-" * 60)

    results = []

    for r_file in resume_files:
        with open(r_file, 'r', encoding='utf-8') as f:
            resume = json.load(f)
            
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
        top_improvements = []
        if missing_skills:
            jd_context = jd_data.get('responsibilities', {}).get('raw_text') or jd_data.get('company_info', {}).get('raw_text') or ""
            jd_context = str(jd_context)
            jd_emb = model.encode(jd_context, convert_to_tensor=True, show_progress_bar=False) if jd_context.strip() else None
            cand_base_context = (resume.get('experience', {}).get('raw_text') or "") + " " + (resume.get('projects') or "")
            req_tech_dict = jd_data.get('required_skills', {}).get('technical', {})
            
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
            top_improvements = top_improvements[:5]
        
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
    lines = []
    lines.append("INTELLIGENT RESUME-JD MATCHING REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {MODEL_NAME}")
    lines.append(f"Scoring: {int(WEIGHT_CONTEXT*100)}% Semantic + {int(WEIGHT_SKILLS*100)}% Skill + {int(WEIGHT_ELIGIB*100)}% Education")
    lines.append("=" * 70)
    
    for rank, r in enumerate(results, 1):
        if r["final_pct"] >= 75:
            verdict = "STRONG MATCH"
        elif r["final_pct"] >= 55:
            verdict = "MODERATE MATCH"
        elif r["final_pct"] >= 35:
            verdict = "DEVELOPING MATCH"
        else:
            verdict = "LOW MATCH"

        lines.append("")
        lines.append(f"RANK #{rank}")
        lines.append(f"Candidate : {r['name']}")
        lines.append(f"Score     : {r['final_pct']}%  [{verdict}]")
        lines.append("")
        
        if r['matched_skills']:
            lines.append("SKILLS YOU HAVE (that the JD wants):")
            lines.append(f"  {', '.join(sorted(r['matched_skills']))}")
            lines.append("")
            
        if r['missing_skills']:
            lines.append("SKILLS YOU ARE MISSING:")
            lines.append(f"  {', '.join(sorted(r['missing_skills']))}")
            lines.append("")
            
        if r['top_improvements']:
            lines.append("YOUR GROWTH PATH (What to Learn Next):")
            for idx, imp in enumerate(r['top_improvements'], 1):
                lines.append(f"  #{idx}  Learn {imp['skill'].title():<22} +{round(imp['impact_pct']*100, 1)}%  ({r['final_pct']}% -> {round(imp['new_pct']*100, 1)}%)")
                
            if len(r['top_improvements']) >= 2:
                total_pot = r['final_pct'] + sum(round(i['impact_pct']*100, 1) for i in r['top_improvements'])
                lines.append("")
                lines.append(f"  Potential score with all above: ~{round(min(total_pot, 99.9), 1)}%")
            lines.append("")
            
        lines.append("PERSONALIZED FEEDBACK:")
        for fb in r['feedback']:
            lines.append(f"  {fb}")
            
        lines.append("-" * 70)

    lines.append("")
    lines.append("QUICK RANKING SUMMARY")
    lines.append("=" * 70)
    lines.append(f"{'Rank':<6} {'Candidate':<40} {'Score':<8} {'Verdict'}")
    lines.append(f"{'-'*6} {'-'*40} {'-'*8} {'-'*15}")
    for rank, r in enumerate(results, 1):
        v = ("Strong" if r["final_pct"] >= 75 else
             "Moderate" if r["final_pct"] >= 55 else
             "Developing" if r["final_pct"] >= 35 else "Low")
        lines.append(f"{rank:<6} {r['name']:<40} {r['final_pct']:<8} {v}")
    lines.append("=" * 70)

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        
    for line in lines:
        print(line)
        
    print(f"\n[OK] Detailed report saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    if run_parser():
        run_matcher()
