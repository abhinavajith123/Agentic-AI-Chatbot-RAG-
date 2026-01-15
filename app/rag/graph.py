from typing import TypedDict, List
from langgraph.graph import StateGraph
from app.rag.retriver import retrive
from app.rag.generator import gen_ans
from app.rag.grounding import is_grounded


class RAGState(TypedDict):
    question: str
    chunks: List[dict]
    answer: str
    confidence: float

# This node does the retreive functionality 
def retrieve_node(state):
    chunks = retrive(state["question"])
    return {"chunks":chunks}

# This node does the Answer generation part
def generate_node(state):
    ans = gen_ans(state["question"], state["chunks"])
    confidence = sum(i["score"] for i in state["chunks"])/ len(state["chunks"])
    return {"answer": ans, "confidence": confidence}

# Called in case of irrelevent question from user
def fallback_node(state):
    return {
        "answer": "I don't know based on the document.",
        "confidence": 0.0
    }

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("fallback", fallback_node)

graph.set_entry_point("retrieve")

def decide_next_node(state):
    if is_grounded(state["chunks"]):
        return "generate"
    else:
        return "fallback"
    
graph.add_conditional_edges(
    "retrieve", #after retrive check condition
    decide_next_node
)


graph.add_edge("generate", "__end__")
graph.add_edge("fallback", "__end__")

rag_app = graph.compile()
