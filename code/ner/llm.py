#!/usr/bin/env python3

import json
import requests
import time
from typing import Dict, List, Optional
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLMExtractor:
    def __init__(self, method="ollama", model_name="mistral:7b"):
        self.method = method
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        if method == "huggingface":
            self.setup_huggingface()
        elif method == "ollama":
            self.setup_ollama()
    
    def setup_ollama(self):
        """Setup Ollama connection"""
        try:
            # Test if Ollama is running
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                logger.info(f"Ollama connected. Using model: {self.model_name}")
            else:
                raise ConnectionError("Ollama not running")
        except Exception as e:
            logger.error(f"Ollama setup failed: {e}")
            logger.info("Make sure to run: ollama serve")
    
    def setup_huggingface(self):
        """Setup Hugging Face transformers"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True  # Reduces memory usage
            )
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Hugging Face setup failed: {e}")

    def create_nanopub_prompt(self, sentence: str) -> str:
        """Create structured prompt for nanopublication extraction"""
        prompt = f"""You are a biomedical knowledge extraction expert. Extract structured information from the given sentence and format it as a nanopublication.

Sentence: "{sentence}"

Extract the following information and return ONLY valid JSON:

{{
  "entities": [
    {{"text": "entity_mention", "type": "GENE|PROTEIN|DRUG|DISEASE|CELL_TYPE|ORGANISM", "start": 0, "end": 10}}
  ],
  "relations": [
    {{"subject": "entity1", "predicate": "INHIBITS|ACTIVATES|REGULATES|TREATS|CAUSES", "object": "entity2", "confidence": "high|medium|low"}}
  ]
}}

Important: Return ONLY the JSON, no additional text."""

        return prompt

    def extract_with_ollama(self, prompt: str) -> Dict:
        """Extract using Ollama API"""
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,  # Low temperature for consistent output
                        'top_p': 0.9,
                        'num_predict': 500
                    }
                }
            )
            print(response)
            if response.status_code == 200:
                result = response.json()
                return self.parse_json_response(result['response'])
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Ollama extraction error: {e}")
            return {}

    def extract_with_huggingface(self, prompt: str) -> Dict:
        """Extract using Hugging Face transformers"""
        try:
            import torch
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=500,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return self.parse_json_response(response)
            
        except Exception as e:
            logger.error(f"Hugging Face extraction error: {e}")
            return {}

    def parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        try:
            # Find JSON in response (sometimes LLMs add extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.debug(f"Raw response: {response}")
            return {}

    def extract_nanopublication(self, sentence: str) -> Dict:
        """Main extraction method"""
        prompt = self.create_nanopub_prompt(sentence)
        
        if self.method == "ollama":
            print("Extracting with Ollama...")
            return self.extract_with_ollama(prompt)
        elif self.method == "huggingface":
            return self.extract_with_huggingface(prompt)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def evaluate_extraction(self, sentence: str, ground_truth: Optional[Dict] = None) -> Dict:
        """Evaluate extraction quality"""
        start_time = time.time()
        result = self.extract_nanopublication(sentence)
        print(result)
        extraction_time = time.time() - start_time
        
        evaluation = {
            'sentence': sentence,
            'extraction_time': extraction_time,
            'entities_found': len(result.get('entities', [])),
            'relations_found': len(result.get('relations', [])),
            'has_context': bool(result.get('context', {})),
            'json_valid': bool(result),  # True if parsing succeeded
            'result': result
        }
        
        if ground_truth:
            # Add comparison metrics if ground truth provided
            evaluation['entity_overlap'] = self.calculate_entity_overlap(
                result.get('entities', []), ground_truth.get('entities', [])
            )
        
        return evaluation

    def calculate_entity_overlap(self, predicted: List[Dict], ground_truth: List[Dict]) -> float:
        """Calculate simple entity text overlap"""
        if not predicted or not ground_truth:
            return 0.0
        
        pred_texts = {entity['text'].lower() for entity in predicted if 'text' in entity}
        true_texts = {entity['text'].lower() for entity in ground_truth if 'text' in entity}
        
        if not true_texts:
            return 0.0
        
        overlap = len(pred_texts.intersection(true_texts))
        return overlap / len(true_texts)


def main():
    """Example usage and testing"""
    # test_sentences = [
    #     "Metformin reduces glucose production in hepatocytes.",
        # "TP53 mutations are found in 60% of colorectal cancers.",
        # "IL-6 levels increased significantly after LPS treatment.",
        # "EGFR inhibitors show promise in treating lung adenocarcinoma.",
        # "p53 does not regulate BRCA1 expression in normal breast tissue."
    # ]

    test_sentences = []
    pubmed_df = pd.read_csv("data/pubmed_papers.tsv", sep="\t")
    for idx, row in pubmed_df.iterrows():
        text = test_sentences.append(f"{row['Title']} {row['Abstract']}")

    # Initialize extractor (change method/model as needed)
    print("Initializing LLM extractor...")
    extractor = LocalLLMExtractor(method="ollama", model_name="mistral:7b")
    
    # Process test sentences
    results = []
    for i, sentence in enumerate(test_sentences):
        print(f"\n--- Processing sentence {i+1} ---")
        print(f"Input: {sentence}")
        
        evaluation = extractor.evaluate_extraction(sentence)
        results.append(evaluation)
        
        print(f"Time: {evaluation['extraction_time']:.2f}s")
        print(f"Entities: {evaluation['entities_found']}")
        print(f"Relations: {evaluation['relations_found']}")
        print(f"Valid JSON: {evaluation['json_valid']}")
        
        if evaluation['result']:
            print("Extracted nanopublication:")
            print(json.dumps(evaluation['result'], indent=2))
    
    # Summary statistics
    print(f"\n--- Summary ---")
    avg_time = sum(r['extraction_time'] for r in results) / len(results)
    valid_extractions = sum(1 for r in results if r['json_valid'])
    total_entities = sum(r['entities_found'] for r in results)
    total_relations = sum(r['relations_found'] for r in results)
    
    print(f"Average extraction time: {avg_time:.2f}s")
    print(f"Valid extractions: {valid_extractions}/{len(results)}")
    print(f"Total entities extracted: {total_entities}")
    print(f"Total relations extracted: {total_relations}")


if __name__ == "__main__":
    main()