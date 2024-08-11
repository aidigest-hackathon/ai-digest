'''
Need to set up the langchain
Need to pull data from bucket
Need to upload data to bucket


Front-end: need to give the data to display
'''
from langchain_fireworks import Fireworks 
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages 

from typing import TypedDict, Annotated
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os 
import openai
from pydantic import BaseModel, Field
import json 

load_dotenv()

# Define the state
class SummaryState(TypedDict):
    research_paper: str
    draft_summary: str
    evaluation: bool 
    date: str
    title: str
    beginner_result: Optional[str]
    intermediate_result: Optional[str]
    advanced_result: Optional[str]
    image_path: Optional[str]
    link: Optional[str]


class GPTSummarizer:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.getenv("FIREWORKS_API_KEY"),
        )
        self.history_context = []
        self.prev_summary = ''
        self.summary_count = 0
        self.entities = []

    def generate(self, state: SummaryState):
        # first generation
        if self.summary_count == 0:
            research_paper = state['research_paper']

            first_summary_prompt = f'''Given the full text of a computer science research paper, we want to generate a summary that is concise yet dense in information.
This will proceed with increasingly helpful and entity-dense summaries of the article.
Step 1. Identify 2-3 informative entities from the paper which are missing from the previously generated summary. 
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the new "Missing Entities". A Missing Entity is:
  - Relevant: to the main story.
  - Specific: descriptive yet concise (5 words or fewer).
  - Novel: not in the previous summary.
  - Faithful: present in the Article.
  - Anywhere: located anywhere in the Article.
We are dealing with the first version of the summary, which should be 5-7 bulletpoint or around ~80 words. Feel free to start with highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
Missing entities can appear anywhere in the new summary. Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
Your final response should be in well-formatted JSON whose keys are "missing_entities" (list) and "summary" (string). There should be no further text or explanation in the response.
The input will be the following format: Input: {{PAPER}}
For example,
_Output_
```json
{{
  "missing_entities": ["entity1", "entity2"],
  "summary": "The authors describe a method of synthesizing data using an external data corpus for improving the diversity. This works well enough, but has drawbacks due to the complexity of the pipeline. The key innovations include a novel sub-sampling procedure as well as a new method for data augmentation."
}}
Now it is your turn, please generate relevant entities and a denser summary.

INPUT: {research_paper}
_Output_'''

            self.history_context.append({
                "role": "user",
                "content": first_summary_prompt,
            })
            chat_completion = self.client.chat.completions.create(
                model="accounts/fireworks/models/mixtral-8x7b-instruct",
                response_format={"type": "json_object"},
                messages=self.history_context
            )
        # refiner
        else:
            research_paper = state['research_paper']
            previous_summary = state['draft_summary']

            refine_summary_prompt = f'''Given the full text of a computer science research paper, we want to generate a summary that is concise yet dense in information.
This will proceed with increasingly helpful and entity-dense summaries of the article.
Step 1. Identify 2-3 informative entities from the paper which are missing from the previously generated summary. 
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the new "Missing Entities". A Missing Entity is:
  - Relevant: to the main story.
  - Specific: descriptive yet concise (5 words or fewer).
  - Novel: not in the previous summary.
  - Faithful: present in the Article.
  - Anywhere: located anywhere in the Article.
We are dealing with the {self.summary_count} version of the summary, which should be 5-7 sentences matching the length of the original summary. Make every word count. Re-write the previous summary to improve flow and make space for additional entities. Make space with fusion, compression, and removal of uninformative phrases like "the article discusses". The summaries should provide commentary and an interpretation of the results, such as the practicality of implementation. Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
Current Entities: {self.entities}
Previous Summary: {previous_summary}
Research Paper: {research_paper}
Your final response should be in well-formatted JSON whose keys are "missing_entities" (list) and "summary" (string). There should be no further text or explanation in the response.
For example,
_Output_
```json
{{
  "missing_entities": ["synthesize", "diversity"],
  "summary": "The authors describe a method of synthesizing data using an external data corpus for improving the diversity. This works well enough, but has drawbacks due to the complexity of the pipeline. The key innovations include a novel sub-sampling procedure as well as a new method for data augmentation."
}}
Now it is your turn, please generate relevant entities and a denser summary.
_Output_'''
            self.history_context.append({
                "role": "user",
                "content": refine_summary_prompt,
            })
            chat_completion = self.client.chat.completions.create(
                model="accounts/fireworks/models/mixtral-8x7b-instruct",
                response_format={"type": "json_object"},
                messages=self.history_context
            )


        return chat_completion.choices[0].message



def generate_response(prompt):
    # TODO: deprecated 
    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.getenv("FIREWORKS_API_KEY"),
    )

    llm = Fireworks(
        api_key= os.getenv("FIREWORKS_API_KEY"),
        model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        max_tokens=256)
    output = llm.invoke(prompt)
    return output 


# Define the nodes
def draft_summary(state: SummaryState) -> SummaryState:
    # Simulate drafting a summary
    full_paper = state["research_paper"]
    prompt = f"""Given the computer science research paper, we want to generate a summary that is concise yet dense in information.
Paper: {full_paper}
This will proceed with increasingly helpful and entity-dense summaries of the article.
Step 1. Identify 2-3 informative entities from the paper which are missing from the previously generated summary. 
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the new "Missing Entities". A Missing Entity is:
  - Relevant: to the main story.
  - Specific: descriptive yet concise (5 words or fewer).
  - Novel: not in the previous summary.
  - Faithful: present in the Article.
  - Anywhere: located anywhere in the Article.
We are dealing with the first version of the summary, which should be 5-7 bulletpoint or around ~80 words. Feel free to start with highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
Missing entities can appear anywhere in the new summary. Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
Your final response should be in well-formatted JSON whose keys are "missing_entities" (list) and "summary" (string). There should be no further text or explanation in the response.
Now it is your turn, please generate relevant entities and a denser summary.
_Output_
    """
    first_summary = generate_response(prompt)
    return {"draft_summary": first_summary}

def refine_summary(state: SummaryState) -> SummaryState:
    # Simulate refining the summary
    prompt = f"""




    refined = f"Refined: {state['draft_summary']}"
    return {"draft_summary": refined}

def evaluate_summary(state: SummaryState) -> SummaryState:
    # Adds a boolean to the field evaluation in SummaryState. True means keep generating, False means good enough 
    evaluation = False  
    #evaluation = f"Evaluation of: {state['refined_summary']}"
    return {"evaluation": evaluation}


def should_continue(state: SummaryState):
    is_continue = state['evaluation']
    if is_continue:
        return "refine"
    else:
        return END



def langchain_driver():
    breakpoint()
    # Create the graph
    workflow = StateGraph(SummaryState)

    # Add nodes
    workflow.add_node("draft", draft_summary)
    workflow.add_node("refine", refine_summary)
    workflow.add_node("evaluate", evaluate_summary)
    

    # Add edges
    workflow.set_entry_point("draft")
    workflow.add_edge("draft", "refine")
    workflow.add_edge("refine", "evaluate")
    workflow.add_conditional_edges("evaluate", should_continue)
    workflow.add_edge("evaluate", "draft")
    #workflow.add_edge("evaluate", END) #TODO: maybe optional

    # Compile the graph
    app = workflow.compile()

    # Run the graph
    result = app.invoke({
        "research_paper": """Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

1Introduction
Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states 
h
t
, as a function of the previous hidden state 
h
t
−
1
 and the input for position 
t
. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

2Background
The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

3Model Architecture
3.2Attention
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

3.2.1Scaled Dot-Product Attention
We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension 
d
k
, and values of dimension 
d
v
. We compute the dot products of the query with all keys, divide each by 
d
k
, and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix 
Q
. The keys and values are also packed together into matrices 
K
 and 
V
. We compute the matrix of outputs as:

3.2.2Multi-Head Attention
Scaled Dot-Product Attention

Refer to caption
Multi-Head Attention

Refer to caption
Figure 2:(left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.
Instead of performing a single attention function with 
d
model
-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values 
h
 times with different, learned linear projections to 
d
k
, 
d
k
 and 
d
v
 dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding 
d
v
-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

MultiHead
. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

3.2.3Applications of Attention in our Model
The Transformer uses multi-head attention in three different ways:

• In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].
• The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
• Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to 
−
∞
) all values in the input of the softmax which correspond to illegal connections. See Figure 2.
3.3Position-wise Feed-Forward Networks
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
"""
    })

    print(result)


def driver():
    langchain_driver()


if __name__ == "__main__":
    driver()