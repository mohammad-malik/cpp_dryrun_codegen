import json
from typing import List, Dict
from rank_bm25 import BM25Okapi
import re
from collections import Counter


class CodeEmbeddings:
    def __init__(self):
        self.questions = []
        self.bm25 = None
        self.tokenized_questions = []
        self.file_type = None

    def detect_file_type(self, data: dict) -> str:
        """Detect if file is OOP or PF format"""
        if "metadata" in data and data["metadata"].get("course_title"):
            if "Object Oriented Programming" in data["metadata"]["course_title"]:
                return "oop"
            elif "Programming Fundamentals" in data["metadata"]["course_title"]:
                return "pf"
        return "unknown"

    def parse_oop_content(self, data: dict) -> List[str]:
        """Parse OOP JSON format"""
        questions = []
        for section in data["sections"]:
            if section["title"] == "Questions":
                for q in section["content"]:
                    questions.append(list(q.values())[0])
        return questions

    def parse_pf_content(self, data: dict) -> List[Dict]:
        """Parse PF JSON format"""
        codes = []
        for section in data["sections"]:
            if "content" in section and "codes" in section["content"]:
                for code_item in section["content"]["codes"]:
                    codes.append(
                        {"context": section["title"], "code": code_item["code"]}
                    )
        return codes

    def load_json(self, json_path: str) -> List[str]:
        """Load and extract content from JSON file"""
        with open(json_path) as f:
            data = json.load(f)

        self.file_type = self.detect_file_type(data)

        if self.file_type == "oop":
            content = self.parse_oop_content(data)
            self.questions = content
            self.tokenized_questions = [self._tokenize(q) for q in content]
        elif self.file_type == "pf":
            content = self.parse_pf_content(data)
            self.questions = content
            self.tokenized_questions = [
                self._tokenize(f"{c['context']} {c['code']}") for c in content
            ]
        else:
            raise ValueError("Unknown file format")

        if self.tokenized_questions:
            self.bm25 = BM25Okapi(self.tokenized_questions)

        return content

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization function"""
        # Convert to lowercase and split on non-alphanumeric characters
        return [word.lower() for word in re.findall(r"\w+", text)]

    def _simple_similarity(self, query: List[str], document: List[str]) -> float:
        """Calculate simple text similarity using word overlap"""
        query_counter = Counter(query)
        doc_counter = Counter(document)
        overlap = sum((query_counter & doc_counter).values())
        return overlap / (len(query) + len(document))

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Hybrid search using simple similarity and BM25"""
        query_tokens = self._tokenize(query)

        # Calculate simple similarity scores
        similarities = []
        for tokens in self.tokenized_questions:
            score = self._simple_similarity(query_tokens, tokens)
            similarities.append(score)

        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Combine and normalize scores
        combined_scores = [
            (i, s + b) for i, (s, b) in enumerate(zip(similarities, bm25_scores))
        ]

        # Sort by combined score
        ranked_indices = [
            i for i, _ in sorted(combined_scores, key=lambda x: x[1], reverse=True)
        ][:k]

        # Format results
        results = []
        seen = set()

        for idx in ranked_indices:
            if self.file_type == "oop":
                result = {"question": self.questions[idx]}
                key = result["question"]
            else:
                result = self.questions[idx]
                key = result["code"]

            if key not in seen and len(results) < k:
                seen.add(key)
                results.append(result)

        return results if results else [{"question": "No results found"}]


if __name__ == "__main__":
    embedder = CodeEmbeddings()
    json_path = "data/pf.json"  # or "data/oop.json"

    content = embedder.load_json(json_path)
    print(f"Loaded {len(content)} items from {embedder.file_type} format")

    query = "What are common programming errors?"
    results = embedder.search(query)

    print(f"\nSearch results for: {query}")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        if embedder.file_type == "oop":
            print(f"Question: {result['question'][:200]}...")
        else:
            print(f"Context: {result['context']}")
            print(f"Code:\n{result['code']}")
