from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
import openai
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv
load_dotenv()
client = Client()

# Define dataset: these are your test cases
dataset_name = "Research Evals Dataset"
dataset_id = "faba1867-6268-410e-a22c-d56def8a87cc"
# dataset = client.create_dataset(dataset_name, description="Research Evals Dataset.")
client.create_examples(
    inputs=[
        {"question": "what can you help with today?"},
        {"question": "is the attached research summary looking good?"},
    ],
    outputs=[
        {"must_mention": ["helpfulness", "Coherence", "Relevance"]},
        {"must_mention": ["helpfulness", "Coherence"]},
    ],
    dataset_id=dataset_id,
)

# Define AI system
openai_client = wrap_openai(openai.Client())

def predict(inputs: dict) -> dict:
    messages = [{"role": "user", "content": inputs["question"]}]
    response = openai_client.chat.completions.create(messages=messages, model="gpt-3.5-turbo")
    return {"output": response}

# Define evaluators
def must_mention(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("output") or ""
    required = example.outputs.get("must_mention") or []
    score = all(phrase in prediction for phrase in required)
    return {"key":"must_mention", "score": score}

experiment_results = evaluate(
    predict, # Your AI system
    data=dataset_name, # The data to predict and grade over
    evaluators=[must_mention], # The evaluators to score the results
    experiment_prefix="research-digest-generator", # A prefix for your experiment names to easily identify them
    metadata={
      "version": "1.0.0",
    },
)

if __name__ == "__main__":
    print(experiment_results)
    print("Done!")