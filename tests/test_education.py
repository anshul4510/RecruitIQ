import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.education_processor import EducationProcessor

def run_tests():
    test_cases = [
        {
            "name": "Case 1: Strict Level Match (B.Tech for Bachelor)",
            "resume": ["B.Tech in CSE"],
            "jd": "Bachelor's degree in Computer Science",
            "expected": 1.0
        },
        {
            "name": "Case 2: Rank-Up Match (Master for Bachelor)",
            "resume": ["M.Tech in Data Science"],
            "jd": "Bachelor's in Engineering",
            "expected": 1.0
        },
        {
            "name": "Case 3: Rank-Down Match (Bachelor for Master)",
            "resume": ["B.E. in Mechanical"],
            "jd": "Master's degree in Mechanical Engineering",
            "expected": 0.5 # (0.3 degree + 0.4 field = 0.7? No, 0.5 * 0.6 + 1.0 * 0.4 = 0.3 + 0.4 = 0.7)
        },
        {
            "name": "Case 4: Related Major Match (CS vs IT)",
            "resume": ["B.Sc in Information Technology"],
            "jd": "B.Tech in Computer Science",
            "expected": 0.88 # (1.0 * 0.6 + 0.7 * 0.4 = 0.6 + 0.28 = 0.88)
        },
        {
            "name": "Case 5: Messy Formats",
            "resume": ["Bachelor of Engineering (Computer Science Engineering)"],
            "jd": "B.Tech CSE preferred",
            "expected": 1.0
        }
    ]

    print("\n--- Running Education Processor Tests ---\n")
    for tc in test_cases:
        score = EducationProcessor.calculate_match_score(tc["resume"], tc["jd"])
        print(f"Test: {tc['name']}")
        print(f"Result: {score} | Expected: ~{tc['expected']}")
        print("-" * 30)

if __name__ == "__main__":
    run_tests()
