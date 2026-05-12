import os

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'Input_pdf')
JD_DIR = os.path.join(INPUT_DIR, 'JD')
RESUME_DIR = os.path.join(INPUT_DIR, 'Resume')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
RESUME_JSON_OUT_DIR = os.path.join(OUTPUT_DIR, 'Resume_Json_Output')
JD_JSON_OUT_DIR = os.path.join(OUTPUT_DIR, 'JD_Json_Output')

JD_PATH = os.path.join(JD_DIR, 'job_description.txt')
PROCESSED_JD_PATH = os.path.join(JD_JSON_OUT_DIR, 'job_profile.json')
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'results_structured.txt')

# Create directories if they don't exist
os.makedirs(JD_DIR, exist_ok=True)
os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs(RESUME_JSON_OUT_DIR, exist_ok=True)
os.makedirs(JD_JSON_OUT_DIR, exist_ok=True)

# ── Model Configuration ──────────────────────────────────────────────
MODEL_NAME = 'all-mpnet-base-v2'
SIM_THRESHOLD = 0.36   # min similarity to classify a line as a section header
MAX_HEADER_LEN = 65    # lines longer than this can't be headers

# ── Scoring Weights ──────────────────────────────────────────────────
WEIGHT_SKILLS   = 0.40  # Direct skill overlap
WEIGHT_CONTEXT  = 0.40  # Semantic match (Projects/Exp vs JD Responsibilities)
WEIGHT_ELIGIB   = 0.20  # Education & Experience match

# ── Education & Skills Configurations ────────────────────────────────
EDU_LEVELS = {
    "PHD": 5, "DOCTORATE": 5,
    "MTECH": 4, "MCA": 4, "MSC": 4, "ME": 4, "MBA": 4, "MASTER": 4, "MASTERS": 4,
    "BTECH": 3, "BCA": 3, "BSC": 3, "BE": 3, "BACHELOR": 3, "BACHELORS": 3,
    "DIPLOMA": 2, "NOT DETECTED": 0
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "problem solving", "adaptability",
    "time management", "critical thinking", "creativity", "collaboration",
    "attention to detail", "emotional intelligence", "project management"
}

SKILL_CATEGORIES = {
    "Programming Languages": {
        "python", "java", "javascript", "typescript", "c++", "c#", "r",
        "scala", "golang", "ruby", "php", "swift", "kotlin", "rust",
        "matlab", "html", "css", "sql", "bash", "shell",
    },
    "Data Science & Machine Learning": {
        "machine learning", "deep learning", "neural networks",
        "natural language processing", "nlp", "computer vision",
        "data science", "data analysis", "data analytics", "data mining",
        "statistical modeling", "predictive modeling", "regression",
        "classification", "clustering", "reinforcement learning",
        "transfer learning", "feature engineering", "model training",
        "supervised learning", "unsupervised learning",
        "speech recognition", "prediction",
    },
    "ML/DL Frameworks & Tools": {
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "opencv", "spacy", "nltk",
        "hugging face", "huggingface", "langchain",
    },
    "AI & Generative AI": {
        "llm", "llms", "large language model", "generative ai",
        "rag", "retrieval augmented generation",
        "vector database", "vector db", "vector dbs",
        "embeddings", "fine-tuning", "prompt engineering",
        "ai agent", "ai agents",
    },
    "Data Engineering & ETL": {
        "etl", "elt", "data pipeline", "data pipelines", "data engineering",
        "data warehousing", "data warehouse", "data lake",
        "pyspark", "spark", "apache spark",
        "kafka", "apache kafka",
        "airflow", "apache airflow",
        "nifi", "apache nifi",
        "talend", "fivetran", "airbyte", "dbt",
        "hadoop", "hive", "mapreduce",
    },
    "Cloud Platforms": {
        "aws", "amazon web services", "azure", "microsoft azure",
        "gcp", "google cloud", "google cloud platform",
        "redshift", "bigquery", "adls", "s3", "ec2", "lambda",
        "sagemaker", "vertex ai", "vertex al", "databricks",
        "snowflake", "cloud computing",
    },
    "BI & Data Visualization": {
        "tableau", "power bi", "powerbi", "qlik", "qlik sense",
        "looker", "domo", "matplotlib", "seaborn", "plotly",
        "data visualization", "dashboard", "dashboards",
        "business intelligence",
    },
    "Databases": {
        "mysql", "postgresql", "postgres", "mongodb", "cassandra",
        "redis", "elasticsearch", "sqlite", "oracle", "sql server",
        "dynamodb", "firebase", "neo4j",
    },
    "DevOps & Infrastructure": {
        "docker", "kubernetes", "jenkins", "ci/cd",
        "git", "github", "gitlab",
        "linux", "terraform", "ansible",
        "rest api", "api", "microservices",
    },
    "Web & App Frameworks": {
        "flask", "django", "fastapi", "spring boot",
        "react", "angular", "vue", "node.js", "nodejs", "express",
        "streamlit", "gradio",
    },
    "Data Libraries": {
        "pandas", "numpy", "scipy",
    },
    "Domain Knowledge": {
        "ecommerce", "telecom", "bfsi", "pharmaceuticals",
        "manufacturing", "media", "sports",
    },
}

ALL_TECH_SKILLS = [(cat, skill) for cat, skills in SKILL_CATEGORIES.items() for skill in skills]

# ── JSON Schemas ─────────────────────────────────────────────────────
RESUME_SCHEMA = {
    "objective_summary": [
        "objective", "summary", "profile", "about me", "professional summary", 
        "career objective", "personal profile", "executive summary", "profile overview", "personal statement"
    ],
    "experience": [
        "experience", "work experience", "employment history", "professional experience", 
        "work history", "career history", "internships", "internship experience", 
        "relevant experience", "industry experience", "freelance experience"
    ],
    "education": [
        "education", "academic background", "academic qualifications", 
        "educational qualifications", "scholastic record"
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "expertise", 
        "technologies", "tools and technologies", "it skills", "proficiencies",
        "areas of expertise", "software skills", "key skills", "skill set", 
        "technology stack", "programming languages", "frameworks and libraries", 
        "databases", "cloud platforms", "devops tools"
    ],
    "projects": [
        "projects", "academic projects", "personal projects", "key projects", 
        "major projects", "technical projects", "industry projects", "capstone project", "open source contributions"
    ],
    "certifications": [
        "certifications", "licenses", "courses", "training", "professional development", 
        "workshops", "online courses", "professional certifications", "training programs"
    ],
    "specialized_it": [
        "machine learning experience", "data science experience", 
        "software development experience", "system design experience", 
        "api development", "backend development", "frontend development", 
        "mobile app development"
    ],
    "achievements": [
        "achievements", "awards", "honors", "accomplishments", "recognition", 
        "scholarships", "medals", "awards and honors", "certifications and achievements", 
        "hackathons and competitions"
    ],
    "research": [
        "research", "publications", "papers", "patents", "conferences", 
        "posters", "presentations", "research experience", "research papers", "technical blogs"
    ],
    "activities": [
        "activities", "extracurricular", "volunteer", "community service", 
        "leadership", "affiliations", "memberships", "hobbies", "interests",
        "leadership experience", "extracurricular activities", "volunteer experience", "languages known"
    ],
    "additional": [
        "conferences and workshops"
    ],
    "references": [
        "references", "declaration"
    ]
}

JD_SCHEMA = {
    "company_info": [
        "about us", "company profile", "who we are", "about the company",
        "our mission", "company overview", "company background", 
        "about the team", "team overview", "organizational structure"
    ],
    "role_summary": [
        "role", "job title", "position", "job summary", "position overview",
        "about the role", "job description", "role name", "job id", "department", 
        "location", "employment type", "work mode", "shift timing", 
        "role overview", "role purpose"
    ],
    "responsibilities": [
        "responsibilities", "duties", "what you will do", "key responsibilities",
        "day to day", "core tasks", "your role", "what you'll do",
        "roles and responsibilities", "duties and responsibilities", 
        "primary responsibilities", "day-to-day responsibilities", 
        "ownership areas", "deliverables"
    ],
    "required_skills": [
        "required skills", "qualifications", "what you need", "requirements",
        "technical skills", "skills and experience", "must have", "competencies",
        "prerequisites", "core competencies", "required technologies", 
        "programming languages", "frameworks and libraries", "tools and technologies", 
        "database knowledge", "cloud technologies", "devops skills",
        "communication skills", "problem solving skills", "team collaboration", 
        "leadership skills", "time management"
    ],
    "good_to_have": [
        "good to have", "nice to have", "preferred skills", "bonus points",
        "desired qualifications", "optional", "plus", "bonus skills", 
        "additional skills", "advantageous experience"
    ],
    "eligibility": [
        "eligibility", "education", "degree", "academic requirements",
        "batch", "passout", "academic criteria", "percentage",
        "required qualifications", "educational requirements", 
        "degree requirements", "certifications required", "eligibility criteria"
    ],
    "experience_required": [
        "experience required", "minimum experience", "years of experience",
        "work experience", "preferred experience", "industry experience", "domain expertise"
    ],
    "benefits": [
        "what we offer", "benefits", "perks", "compensation", "salary",
        "ctc", "remuneration", "working conditions", "salary range", 
        "compensation package", "perks and incentives"
    ],
    "domains": [
        "domains", "industries", "verticals", "business areas"
    ],
    "process": [
        "process", "interview process", "selection process", "hiring process",
        "rounds", "recruitment process", "assessment process"
    ],
    "expectations": [
        "kpis", "success metrics", "performance expectations", 
        "goals and objectives", "growth expectations"
    ],
    "work_environment": [
        "work culture", "team dynamics", "reporting structure", "stakeholder interaction"
    ],
    "legal_closing": [
        "equal opportunity statement", "compliance requirements", 
        "background check", "notice period", "how to apply"
    ]
}
