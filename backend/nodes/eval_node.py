from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate, LangChainStringEvaluator
import openai
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Define AI system
openai_client = wrap_openai(openai.Client())


def generate_results(inputs: str) -> dict:
    # TODO: call function to generat the summary
    summary = "This is a summary of the research paper"
    return {"output": summary}


def evaluate_summary(inputs):

    client = Client()
    dataset_name = "Research Paper Summary" + datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset = client.create_dataset(dataset_name=dataset_name)

    client.create_examples(inputs=inputs, dataset_id=dataset.id)

    brevity_evaluator = LangChainStringEvaluator(
        "score_string",
        config={
            "criteria": {
                "brevity": "How concise, coherent and well-structured is the summary on a scale of 1-10? It should be no more than 7 sentences." 
            },
            "normalize_by": 10,
        }  
    )

    fluency_evaluator = LangChainStringEvaluator(
        "score_string",
        config={
            "criteria": {
                "fluency": "Can the summary be understood in isolation, without reference to the original paper on a scale of 1-10?" 
            },
            "normalize_by": 10,
        }  
    )

    repeatness_evaluator = LangChainStringEvaluator(
        "score_string",
        config={
            "criteria": {
                "repeatness": "Is the summary unique and does it paraphrase the abstract instead of just repeating the words on a scale of 1-10?" 
            },
            "normalize_by": 10,
        }  
    )


    results = evaluate(
        generate_results,
        data=dataset_name,
        evaluators=[brevity_evaluator, fluency_evaluator, repeatness_evaluator],
        experiment_prefix="research-digest-generator", # A prefix for your experiment names to easily identify them
        metadata={
        "version": "1.0.0",
        },
    )

    for result in results:
        for eval_result in result["evaluation_results"]:
            print(eval_result.get("key"), eval_result.get("score"))


