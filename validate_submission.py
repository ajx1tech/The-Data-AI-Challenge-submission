import csv
import re
import sys

def validate(csv_path: str):
    errors = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        expected_header = ['candidate_id', 'rank', 'score', 'reasoning']
        if header != expected_header:
            errors.append(f"Header mismatch. Expected {expected_header}, got {header}")
            
        rows = list(reader)
        
    if len(rows) != 100:
        errors.append(f"Row count mismatch. Expected exactly 100 rows, got {len(rows)}")
        
    ranks = []
    ids = set()
    prev_score = float('inf')
    prev_id = ""
    
    for i, row in enumerate(rows):
        try:
            cid, rank_str, score_str, reasoning = row
        except ValueError:
            errors.append(f"Row {i+1}: Malformed columns. Expected 4 columns.")
            continue
            
        rank = int(rank_str)
        score = float(score_str)
        ranks.append(rank)
        
        # Validate ID pattern
        if not re.match(r'^CAND_\d{7}$', cid):
            errors.append(f"Row {i+1}: Invalid candidate_id format: '{cid}'")
            
        # Validate uniqueness
        if cid in ids:
            errors.append(f"Row {i+1}: Duplicate candidate_id: {cid}")
        ids.add(cid)
        
        # Validate non-increasing rule
        if score > prev_score:
            errors.append(f"Row {i+1}: Score strictly increasing! {score} > {prev_score}")
            
        # Validate tie-breaking logic
        if score == prev_score:
            if cid < prev_id:
                errors.append(f"Row {i+1}: Tie-breaking failed. ID '{cid}' should not come before '{prev_id}' lexicographically.")
                
        # Validate reasoning length
        if len(reasoning) > 200:
            errors.append(f"Row {i+1}: Reasoning length {len(reasoning)} exceeds max 200 characters.")
            
        prev_score = score
        prev_id = cid
        
    # Validate ranks are exactly 1-100 without duplicates or gaps
    expected_ranks = list(range(1, 101))
    if sorted(ranks) != expected_ranks:
        errors.append("Ranks must be exactly 1 to 100.")
        
    if errors:
        print("❌ VALIDATION FAILED with the following errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✅ VALIDATION PASSED! Submission structure is perfect.")
        sys.exit(0)

if __name__ == "__main__":
    validate('output/submission.csv')
