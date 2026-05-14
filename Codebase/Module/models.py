from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# --- Sub-models ---

class MetaInfo(BaseModel):
    source_file: str
    processed_at: str
    sections_detected: List[str]

class PersonalInfo(BaseModel):
    name: str = "Unknown"
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    huggingface: Optional[str] = None

class SkillSet(BaseModel):
    technical: Dict[str, List[str]] = Field(default_factory=dict)
    soft: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None

class EducationInfo(BaseModel):
    raw_text: Optional[str] = None
    highest_degree: str = "NOT DETECTED"
    certifications: List[str] = Field(default_factory=list)

class ExperienceInfo(BaseModel):
    raw_text: Optional[str] = None
    years_detected: int = 0

class CompanyInfo(BaseModel):
    name: str = "Unknown"
    location: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None

class RoleSummary(BaseModel):
    title: str = "Unknown"
    raw_text: Optional[str] = None

class RequirementsInfo(BaseModel):
    key_points: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None

class EligibilityInfo(BaseModel):
    batch_year: Optional[str] = None
    degrees_required: List[str] = Field(default_factory=list)
    min_percentage: Optional[str] = None
    raw_text: Optional[str] = None

class BenefitsInfo(BaseModel):
    ctc: Optional[str] = None
    raw_text: Optional[str] = None

# --- Main Models ---

class CandidateProfile(BaseModel):
    meta: MetaInfo
    personal_info: PersonalInfo
    objective_summary: Optional[str] = None
    skills: SkillSet
    experience: ExperienceInfo
    projects: Optional[str] = None
    specialized_it: Optional[str] = None
    education: EducationInfo
    certifications: Optional[str] = None
    achievements: Optional[str] = None
    research: Optional[str] = None
    activities: Optional[str] = None
    additional_sections: Optional[str] = None
    references: Optional[str] = None

class JobDescription(BaseModel):
    meta: MetaInfo
    company_info: CompanyInfo
    role_summary: RoleSummary
    responsibilities: RequirementsInfo
    required_skills: SkillSet
    good_to_have: SkillSet
    eligibility: EligibilityInfo
    experience_required: ExperienceInfo
    benefits: BenefitsInfo
    domains: List[str] = Field(default_factory=list)
    process: Optional[str] = None
    expectations: Optional[str] = None
    work_environment: Optional[str] = None
    legal_closing: Optional[str] = None

class MatchResult(BaseModel):
    name: str
    final_pct: float
    elig_pct: float
    skill_pct: float
    context_pct: float
    cand_degree: str
    matched_skills: List[str]
    missing_skills: List[str]
    top_improvements: List[Dict[str, Any]]
    feedback: List[str]
