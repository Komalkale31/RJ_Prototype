from datetime import datetime
from .config import MODEL_NAME, WEIGHT_CONTEXT, WEIGHT_SKILLS, WEIGHT_ELIGIB

def get_category_for_skill(skill, req_tech_dict):
    for cat, skills in req_tech_dict.items():
        if skill in skills:
            return cat
    return "Technical Domain"

def generate_smart_feedback(final_score, skill_score, context_score, elig_score, cand_tech_dict, req_tech_dict, matched_skills, missing_skills, elig_feedback, top_improvements):
    lines = []
    
    # 0. Score Breakdown
    lines.append("SCORE BREAKDOWN:")
    lines.append(f"  - Technical Skills  : {round(skill_score*100, 1):>5}% (Weight 40%)")
    lines.append(f"  - Project Context   : {round(context_score*100, 1):>5}% (Weight 40%)")
    lines.append(f"  - Eligibility       : {round(elig_score*100, 1):>5}% (Weight 20%)")
    lines.append("-" * 70)
    
    # Verdict
    if final_score >= 0.75:
        lines.append("You are a STRONG fit for this role. Your profile demonstrates solid alignment with the job requirements.")
    elif final_score >= 0.55:
        lines.append("You are a MODERATE fit for this role. You have a good foundation but there are specific areas where upskilling would significantly improve your chances.")
    elif final_score >= 0.35:
        lines.append("You are currently a DEVELOPING fit for this role. While your background shows some relevance, focused skill development in key areas is needed to become competitive.")
    else:
        lines.append("Your profile shows LIMITED alignment with this role at present. Significant upskilling and project experience in the required domains would be needed.")
        
    # Eligibility Warning
    if elig_feedback:
        lines.append(f"WARNING: Eligibility Concerns -> {' | '.join(elig_feedback)}")

    # Strength
    if cand_tech_dict:
        strongest_cat = max(cand_tech_dict, key=lambda k: len(cand_tech_dict[k]))
        strong_skills = cand_tech_dict[strongest_cat]
        lines.append(f"YOUR STRENGTH: Your strongest area is '{strongest_cat}' - specifically {', '.join(sorted(strong_skills)[:4])}. This gives you a solid technical base.")
        
    # Key Gap
    if missing_skills:
        missing_by_cat = {}
        for skill in missing_skills:
            cat = get_category_for_skill(skill, req_tech_dict)
            missing_by_cat[cat] = missing_by_cat.get(cat, 0) + 1
            
        weakest_cat = max(missing_by_cat, key=missing_by_cat.get)
        weak_count = missing_by_cat[weakest_cat]
        lines.append(f"KEY GAP: The biggest gap in your profile is in '{weakest_cat}' where you are missing {weak_count} skills that the JD specifically asks for.")

    # Highest Impact Action (What-If)
    if top_improvements:
        top = top_improvements[0]
        lines.append(
            f"HIGHEST IMPACT ACTION: If you learn '{top['skill'].title()}', "
            f"your match score is projected to jump from {round(final_score*100, 1)}% -> {round(top['new_pct']*100, 1)}% "
            f"(+{round(top['impact_pct']*100, 1)}%). This skill falls under '{top['category']}'."
        )

        if len(top_improvements) >= 3:
            second = top_improvements[1]
            third = top_improvements[2]
            lines.append(
                f"NEXT STEPS: After that, focus on '{second['skill'].title()}' "
                f"(+{round(second['impact_pct']*100, 1)}%) and '{third['skill'].title()}' "
                f"(+{round(third['impact_pct']*100, 1)}%). Together, these three skills would make you a significantly stronger candidate."
            )

    # Context Tip
    if context_score < 0.40 and (len(matched_skills) / max(1, len(missing_skills + matched_skills))) > 0.5:
        lines.append("NOTE: You have many of the right technical skills, but your resume may not be contextualized well for this specific role. Consider rewriting your project descriptions to highlight how your skills apply to the type of work described in this JD.")
    elif context_score > 0.60 and (len(matched_skills) / max(1, len(missing_skills + matched_skills))) < 0.4:
        lines.append("NOTE: Your overall experience 'sounds' relevant to this role, but you are missing many specific tools and technologies mentioned in the JD. Focus on hands-on projects with those exact tools to bridge this gap.")
        
    return lines

def generate_full_report(results):
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
    
    return "\n".join(lines)
