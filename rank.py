import gzip
import json
import faiss
import numpy as np
import datetime
import csv
import os
import sys
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class ReasoningGenerator:
    def get_years_exp(self, candidate: dict) -> float:
        experiences = candidate.get('work_experience', [])
        total_days = 0
        for exp in experiences:
            try:
                start = datetime.datetime.strptime(exp.get('start_date', '')[:10], '%Y-%m-%d')
                end_str = exp.get('end_date')
                end = datetime.datetime.strptime(end_str[:10], '%Y-%m-%d') if end_str else datetime.datetime.now()
                total_days += max(0, (end - start).days)
            except (ValueError, TypeError):
                continue
        return round((total_days / 365.25), 1)

    def generate(self, rank: int, candidate: dict) -> str:
        title = candidate.get('title', 'AI Engineer')
        yrs = self.get_years_exp(candidate)
        
        skills = candidate.get('skills', [])
        skill_names = [s.get('name', '') for s in skills if isinstance(s, dict) and s.get('name')]
        top_3_skills = ", ".join(skill_names[:3]) if skill_names else "general skills"
        
        signals = candidate.get('redrob_signals', {})
        resp_rate = int(signals.get('recruiter_response_rate', 0.0) * 100)
        notice_days = signals.get('notice_period_days', 999)
        city = candidate.get('location', {}).get('city', 'Unknown')
        
        if rank <= 30:
            reason = f"{title} with {yrs} yrs; strong in {top_3_skills}; {resp_rate}% response rate; {city}-based."
        elif rank <= 60:
            strengths = f"strong in {top_3_skills}"
            if resp_rate < 50:
                concern = f"low response ({resp_rate}%)"
            elif notice_days > 60:
                concern = f"long notice ({notice_days}d)"
            else:
                concern = "no major concerns"
            reason = f"{title} with {yrs} yrs; {strengths}; {concern}."
        else:
            if resp_rate < 30:
                concern = f"low response ({resp_rate}%)"
            elif notice_days > 60:
                concern = f"long notice ({notice_days}d)"
            elif 'python' not in " ".join(skill_names).lower():
                concern = "missing core python"
            else:
                concern = "logistical mismatch"
            reason = f"{title} with {yrs} yrs; concern: {concern}."
            
        return reason[:200]

class CandidateScorer:
    def __init__(self):
        self.consulting_companies = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}
        self.irrelevant_titles = {'marketing', 'accountant', 'content writer', 'graphic designer', 
                                  'hr manager', 'sales executive', 'customer support', 
                                  'operations manager', 'business analyst', 'project manager', 
                                  'mechanical engineer', 'civil engineer'}
        self.ml_keywords = {'ml', 'ai', 'data', 'machine learning', 'deep learning', 'nlp'}
        self.evaluation_date = datetime.datetime(2024, 7, 1)

    def _parse_date(self, date_str: str) -> datetime.datetime:
        if not date_str:
            return None
        try:
            return datetime.datetime.strptime(date_str[:10], '%Y-%m-%d')
        except ValueError:
            try:
                return datetime.datetime.strptime(date_str[:7], '%Y-%m')
            except ValueError:
                return None

    def is_honeypot(self, candidate: dict) -> bool:
        for exp in candidate.get('work_experience', []):
            start = self._parse_date(exp.get('start_date'))
            founded = self._parse_date(exp.get('company_founded_date'))
            if start and founded and start < founded:
                return True
                
        skills = candidate.get('skills', [])
        mismatch_count = 0
        excessive_count = 0
        
        for skill in skills:
            prof = str(skill.get('proficiency', '')).lower()
            years = int(skill.get('years_used', 0))
            
            if prof in ['expert', 'advanced']:
                excessive_count += 1
                if years == 0:
                    mismatch_count += 1
                    
        if mismatch_count >= 3 or excessive_count > 8:
            return True
            
        return False

    def _get_disqualifier_multipliers(self, candidate: dict) -> float:
        multiplier = 1.0
        title = str(candidate.get('title', '')).lower()
        if any(irrel in title for irrel in self.irrelevant_titles):
            multiplier *= 0.02
            
        experiences = candidate.get('work_experience', [])
        if experiences:
            consulting_count = sum(
                1 for exp in experiences 
                if any(cons in str(exp.get('company', '')).lower() for cons in self.consulting_companies)
            )
            if consulting_count == len(experiences):
                multiplier *= 0.05
                
        skills_text = " ".join([str(s.get('name', '')).lower() for s in candidate.get('skills', [])])
        if 'python' not in skills_text:
            multiplier *= 0.10
            
        return multiplier

    def _score_skills(self, candidate: dict) -> float:
        skills_text = " ".join([str(s.get('name', '')).lower() for s in candidate.get('skills', [])])
        score = 0.0
        
        if 'python' in skills_text: score += 0.10
        if 'embed' in skills_text: score += 0.15
        if 'vector' in skills_text: score += 0.15
        if 'eval' in skills_text: score += 0.10
        if 'rank' in skills_text: score += 0.10
        if 'retriev' in skills_text: score += 0.10
        if 'search' in skills_text: score += 0.08
        if 'recommend' in skills_text: score += 0.07
        if 'production' in skills_text: score += 0.08
        if 'faiss' in skills_text: score += 0.07
        
        return min(1.0, score)

    def _score_experience(self, candidate: dict) -> float:
        experiences = candidate.get('work_experience', [])
        if not experiences:
            return 0.0
            
        total_days = 0
        consulting_count = 0
        ml_count = 0
        has_production = False
        
        for exp in experiences:
            start = self._parse_date(exp.get('start_date'))
            end = self._parse_date(exp.get('end_date'))
            
            if start:
                end_date = end if end else self.evaluation_date
                total_days += max(0, (end_date - start).days)
                
            comp = str(exp.get('company', '')).lower()
            if any(cons in comp for cons in self.consulting_companies):
                consulting_count += 1
                
            combined_text = (str(exp.get('title', '')) + " " + str(exp.get('description', ''))).lower()
            
            if any(kw in combined_text for kw in self.ml_keywords):
                ml_count += 1
                
            if 'production' in combined_text or 'shipped' in combined_text:
                has_production = True
                
        total_years = total_days / 365.25
        score = 0.0
        
        if 6 <= total_years <= 8:
            score += 1.0
        elif 5 <= total_years <= 9:
            score += 0.7
            
        ml_ratio = ml_count / len(experiences) if len(experiences) > 0 else 0
        score += (0.2 * ml_ratio)
        
        if has_production:
            score += 0.15
            
        consulting_ratio = consulting_count / len(experiences) if len(experiences) > 0 else 0
        if consulting_ratio < 1.0:
            score += 0.10
            
        score -= (0.2 * consulting_ratio)
        
        return max(0.0, min(1.0, score))

    def _score_behavioral(self, candidate: dict) -> float:
        signals = candidate.get('redrob_signals', {})
        score = 0.0
        
        if signals.get('recruiter_response_rate', 0.0) >= 0.7:
            score += 0.30
            
        last_active = str(signals.get('last_active_date', ''))
        if '2024' in last_active or '2025' in last_active:
            score += 0.25
            
        open_to_work = signals.get('open_to_work_flag', False)
        applied_recently = signals.get('applied_recently', False)
        if open_to_work and applied_recently:
            score += 0.20
            
        if signals.get('github_activity_score', 0) >= 50:
            score += 0.10
            
        return min(1.0, score)

    def _score_logistics(self, candidate: dict) -> float:
        location = candidate.get('location', {}).get('city', '').lower()
        signals = candidate.get('redrob_signals', {})
        score = 0.0
        
        if any(city in location for city in ['pune', 'noida', 'delhi']):
            score += 0.60
        elif any(city in location for city in ['mumbai', 'bangalore', 'bengaluru', 'hyderabad']):
            score += 0.40
            
        if signals.get('notice_period_days', 999) <= 30:
            score += 0.40
            
        return min(1.0, score)

    def _get_red_flag_penalty(self, candidate: dict) -> float:
        signals = candidate.get('redrob_signals', {})
        penalty = 0.0
        
        if signals.get('recruiter_response_rate', 1.0) < 0.2:
            penalty += 0.15
            
        last_active = self._parse_date(signals.get('last_active_date'))
        if last_active:
            days_inactive = (self.evaluation_date - last_active).days
            if days_inactive > 180:
                penalty += 0.10
        else:
            penalty += 0.10
            
        if signals.get('notice_period_days', 0) > 90:
            penalty += 0.08
            
        experiences = candidate.get('work_experience', [])
        if len(experiences) > 0:
            total_days = 0
            for exp in experiences:
                start = self._parse_date(exp.get('start_date'))
                end = self._parse_date(exp.get('end_date'))
                if start:
                    end_date = end if end else self.evaluation_date
                    total_days += max(0, (end_date - start).days)
                    
            avg_tenure_years = (total_days / len(experiences)) / 365.25
            if avg_tenure_years < 1.5:
                penalty += 0.10
        else:
            penalty += 0.10
            
        return min(0.4, penalty)

    def calculate_score(self, candidate: dict) -> float:
        if self.is_honeypot(candidate):
            return 0.0
            
        skills_score = self._score_skills(candidate)
        exp_score = self._score_experience(candidate)
        behav_score = self._score_behavioral(candidate)
        log_score = self._score_logistics(candidate)
        
        base_score = (
            (skills_score * 0.35) +
            (exp_score * 0.25) +
            (behav_score * 0.25) +
            (log_score * 0.15)
        )
        
        base_score *= self._get_disqualifier_multipliers(candidate)
        red_flag_penalty = self._get_red_flag_penalty(candidate)
        final_score = base_score - red_flag_penalty
        
        return max(0.0, final_score)

def load_candidates(filepath: str) -> list[dict]:
    candidates = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading candidates"):
            candidates.append(json.loads(line))
    return candidates

def get_candidate_text(c: dict) -> str:
    title = c.get('title', '') or ''
    summary = c.get('summary', '') or ''
    
    skills = c.get('skills', [])
    skill_names = [s.get('name', '') for s in skills if isinstance(s, dict) and s.get('name')]
    skills_text = ", ".join(skill_names)
    
    experience = c.get('work_experience', [])
    exp_texts = []
    for exp in experience[:4]:
        exp_title = exp.get('title', '') or ''
        exp_company = exp.get('company', '') or ''
        exp_desc = exp.get('description', '') or ''
        exp_texts.append(f"{exp_title} at {exp_company}: {exp_desc}")
    
    exp_combined = " | ".join(exp_texts)
    parts = [title, summary, skills_text, exp_combined]
    return " ".join(p for p in parts if p.strip())

def prepare_jd_text(jd_text: str) -> str:
    keywords = ['require', 'skill', 'experience']
    extracted_lines = []
    for line in jd_text.splitlines():
        if any(kw in line.lower() for kw in keywords):
            extracted_lines.append(line.strip())
            
    extracted_text = " ".join(extracted_lines)
    first_1500_chars = jd_text[:1500]
    return f"{first_1500_chars}\n\nKey Requirements extracted:\n{extracted_text}".strip()

def generate_embeddings(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings

def build_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def main():
    jd_path = 'job_description.md'
    data_path = 'data/candidates.jsonl.gz'
    output_path = 'output/submission.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found. Please place it in the data/ folder.")
        sys.exit(1)
        
    jd_text = ""
    if os.path.exists(jd_path):
        with open(jd_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    else:
        print(f"Warning: {jd_path} not found. Using default text.")
        jd_text = "Senior AI Engineer - Founding Team. Require Python, vector databases, embeddings, evaluation, ranking, retrieval."

    print("1. Loading candidates...")
    candidates = load_candidates(data_path)
    
    print("2. Extracting text for embeddings...")
    candidate_texts = [get_candidate_text(c) for c in candidates]
    
    print("3. Generating embeddings...")
    embeddings = generate_embeddings(candidate_texts)
    
    print("4. Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    print("5. Semantic Retrieval (Top 500)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    prepared_jd = prepare_jd_text(jd_text)
    jd_embedding = model.encode([prepared_jd], normalize_embeddings=True)
    
    D, I = index.search(jd_embedding, 500)
    top_500_indices = I[0]
    
    print("6. Filtering Honeypots & Scoring...")
    scorer = CandidateScorer()
    scored_candidates = []
    
    for idx in top_500_indices:
        cand = candidates[idx]
        if scorer.is_honeypot(cand):
            continue
            
        score = scorer.calculate_score(cand)
        scored_candidates.append({
            'candidate': cand,
            'score': score
        })
        
    print(f"Candidates remaining after honeypot filtering: {len(scored_candidates)}")
    
    scored_candidates.sort(key=lambda x: (-x['score'], x['candidate'].get('candidate_id', '')))
    
    top_100 = scored_candidates[:100]
    if len(top_100) < 100:
        print(f"Warning: Only {len(top_100)} candidates left after filtering. Needed 100.")
    
    print("7. Generating Reasoning & Writing to CSV...")
    os.makedirs('output', exist_ok=True)
    generator = ReasoningGenerator()
    prev_score = float('inf')
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['candidate_id', 'rank', 'score', 'reasoning']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, item in enumerate(top_100):
            rank = i + 1
            cand = item['candidate']
            score = item['score']
            
            if score > prev_score:
                score = prev_score - 0.0001
                
            reasoning = generator.generate(rank, cand)
            
            writer.writerow({
                'candidate_id': cand.get('candidate_id', f'UNKNOWN_{i}'),
                'rank': rank,
                'score': round(score, 5),
                'reasoning': reasoning
            })
            
            prev_score = score

    print(f"Successfully wrote top 100 candidates to {output_path}")

if __name__ == "__main__":
    main()
