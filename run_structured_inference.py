from __future__ import annotations
import json, argparse, yaml
from pathlib import Path
from typing import Dict, Any, Iterator
from tqdm import tqdm
from models import make_model
from data_utils import load_taxonomy

def extract_json(text: str) -> str:
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

def build_structured_queries(
    dataset, dataset_path, semantic_prompt_path, eval_prompt_path, taxonomy_path, output_path, has_gt=True
):
    querys = []
    existing_ids = set()
    if Path(output_path).exists():
        existing_ids = {json.loads(line).get("answer_id") for line in open(output_path, "r", encoding="utf-8") if line.strip()}
    
    questions_dict = {}
    for question in dataset['questions']:
        question_id = question['question_id']
        questions_dict[question_id] = question

    with open(semantic_prompt_path, "r", encoding="utf-8") as f:
        semantic_prompts = yaml.safe_load(f)
    
    with open(eval_prompt_path, "r", encoding="utf-8") as f:
        eval_prompts = yaml.safe_load(f)

    taxonomy = load_taxonomy(taxonomy_path)

    for annotation in dataset['annotations']:
        answer_id = annotation['answer_id']
        if answer_id in existing_ids:
            continue
            
        question_id = annotation['question_id']
        has_image = questions_dict[question_id].get('requires_input_image', True)
        question_text = questions_dict[question_id]['query_en']
        # Default to "ALL" taxonomy if category not found (for robustness)
        cat = annotation.get('category', 'ALL')
        taxonomy_text = taxonomy.get(cat, taxonomy.get('ALL', ''))

        # Prepare base sample context
        sample_base = {
            'answer_id': answer_id,
            'question_id': question_id,
        }
        
        # Determine prompt key based on inputs
        if has_image:
            sample_base['image_0'] = str(dataset_path / questions_dict[question_id]['input_image_path'])
            if has_gt:
                prompt_key = 'image_with_gt'
                sample_base['image_1'] = str(dataset_path / questions_dict[question_id]['gt_image_path'])
                sample_base['image_2'] = str(dataset_path / annotation['image_path'])
            else:
                prompt_key = 'image_no_gt'
                sample_base['image_1'] = str(dataset_path / annotation['image_path'])
        else:
            if has_gt:
                prompt_key = 'no_image_with_gt'
                sample_base['image_0'] = str(dataset_path / questions_dict[question_id]['gt_image_path'])
                sample_base['image_1'] = str(dataset_path / annotation['image_path'])
            else:
                prompt_key = 'no_image_no_gt'
                sample_base['image_0'] = str(dataset_path / annotation['image_path'])

        # Build Semantic Extraction Step Sample
        sem_sample = dict(sample_base)
        sem_query = semantic_prompts[prompt_key].replace("[[QUERY_EN]]", question_text)
        sem_sample['query'] = sem_query

        # Build Eval Step Base Sample (will inject STUDENT_SEMANTICS later)
        eval_sample = dict(sample_base)
        eval_query_template = eval_prompts[prompt_key].replace("[[QUERY_EN]]", question_text)
        eval_query_template = eval_query_template.replace("[[TAXONOMY_BULLETED]]", taxonomy_text)
        
        # We store the template to format it after Step 1 completes
        querys.append({
            'answer_id': answer_id,
            'sem_sample': sem_sample,
            'eval_sample_base': eval_sample,
            'eval_query_template': eval_query_template
        })

    return querys

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', type=str, default='', help="local model path or huggingface model name")
    p.add_argument('--semantic-prompt-path', type=str, default='./prompts/prompt_semantic_extraction.yaml')
    p.add_argument('--eval-prompt-path', type=str, default='./prompts/prompt_structured_eval.yaml')
    p.add_argument("--model-type", default="qwen", help="key in MODEL_REGISTRY")
    p.add_argument("--model-name", type=str, default='', help="for GPT model, the model name like gpt-4o")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output", required=True, help="destination JSONL")
    p.add_argument("--with-reference", action="store_true", default=True, help="Use reference answers (default)")
    p.add_argument("--no-reference", action="store_false", dest="with_reference", help="Disable reference answers")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument('--api-key', type=str, default='')
    args = p.parse_args()

    if args.model_type == "gpt":
        model = make_model(
            args.model_type,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            api_key=args.api_key,
        )
    else:
        model = make_model(
            args.model_type,
            model_path=args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    master_file = args.dataset_path + "/master.json"
    taxonomy_file = args.dataset_path + "/biology_taxonomy.json"

    with open(master_file, "r", encoding="utf-8") as data_file:
        dataset = json.load(data_file)
        print("Dataset loaded.")

    queries = build_structured_queries(
        dataset,
        dataset_path=dataset_path,
        semantic_prompt_path=args.semantic_prompt_path,
        eval_prompt_path=args.eval_prompt_path,
        taxonomy_path=taxonomy_file,
        output_path=out_path,
        has_gt=args.with_reference
    )

    with open(out_path, "a", encoding="utf-8") as fout:
        for item in tqdm(queries, desc="Structured Inference"):
            sem_sample = item['sem_sample']
            eval_sample = item['eval_sample_base']
            eval_query_template = item['eval_query_template']
            
            # Step 1: Extract Semantics
            raw_semantics = model.generate_from_sample(sem_sample)
            extracted_json = extract_json(raw_semantics)
            
            # Step 2: Evaluate
            eval_query = eval_query_template.replace("[[STUDENT_SEMANTICS]]", extracted_json)
            eval_sample['query'] = eval_query
            raw_eval = model.generate_from_sample(eval_sample)
            
            # Save results
            final_record = {
                "answer_id": item['answer_id'],
                "extracted_semantics": extracted_json,
                "response": extract_json(raw_eval),
                "raw_semantics_output": raw_semantics,
                "raw_eval_output": raw_eval
            }
            
            fout.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            fout.flush()
            
            # Optional sleep to avoid rate limits
            import time
            time.sleep(1.0)

    print("✓ Done – results saved to", out_path.resolve())

if __name__ == "__main__":
    main()
