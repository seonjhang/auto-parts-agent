import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from difflib import SequenceMatcher


embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the whole LangChain vectorstore (index + metadata), not raw faiss.read_index
db = FAISS.load_local("my_faiss", embeddings_model, allow_dangerous_deserialization=True)

# Graph State
class AgentState(TypedDict):
    query: str
    intent: dict
    is_ambiguous: bool
    candidates: List[dict]
    ranked: List[dict]
    response: str

llm = ChatOllama(model="llama3.2", temperature=0)

EXTRACTION_PROMPT = """You are a query understanding agent for an auto parts search system.
Extract strcutred information from the user's query and respond with ONLY vlaid JSON, no other text.

Fields to extract:
- "category": the part category if mentioned (e.g. "Brakes", "Engine"), else null
- "part": specific part name if mentioned (e.g. "brake pads"), else null
- "make": car brand if mentioned (e.g. "Honda"), else null
- "model": car model if mentioned (e.g. "Civic"), else null
- "year": model year if mentioned (e.g. 2018), else null
- "is_ambiguous": true if the query is missing both a part/category AND a make+model (not enough info to search), else false

User query: "{query}"

JSON:"""

# --- Nodes ---

def query_understanding(state: AgentState) -> AgentState:
    """First LLM call to extract the intent of the query"""
    prompt = EXTRACTION_PROMPT.format(query=state["query"])
    response = llm.invoke(prompt)

    try:
        intent = json.loads(response.content)
    except json.JSONDecodeError:
        # LLM didn't return clean JSON - fail safe by treating it as ambigfuous
        intent = {"category": None, "part": None, "make": None, "model": None, "year": None}
        intent["is_ambiguous"] = True
    
    return {
        "intent": intent,
        "is_ambiguous": intent.get("is_ambiguous", False)
    }

def route_after_understanding(state: AgentState) -> str:
    return "clarify" if state["is_ambiguous"] else "retrieval"

def clarify(state: AgentState) -> AgentState:
    return {"response": "I can help you find the right part if you clarify the car make, model, or year."}


def retrieval(state: AgentState, k: int = 5) -> AgentState:
    """Search the FAISS vectorstore for the top-k most similar parts"""
    results = db.similarity_search(state["query"], k=k)

    candidates = [doc.metadata for doc in results]

    print(f"\nUser query: '{state['query']}'")
    print("=" * 50)
    for c in candidates:
        print(f"- {c.get('part_name')} | {c.get('compatible_make')} {c.get('compatible_model')} | ${c.get('price_usd')}")

    return {"candidates": candidates}

def ranking(state: AgentState) -> AgentState:
    
    intent = state["intent"]
    candidates = state["candidates"]

    scored = []

    for c in candidates:
        s = score_candidate(intent, c)
        scored.append((s, c))
    
    score_sorted= sorted(scored, key=lambda x: x[0], reverse=True)
    
    candidates = [t[1] for t in score_sorted]
    
    return {"ranked": candidates}

WEIGHTS = {
    "category": 1.0,
    "make": 2.0,
    "model": 2.0,
    "year": 1.5
}
MAX_SCORE = sum(WEIGHTS.values())

def text_similarity(a, b) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def score_candidate(intent: dict, candidate: dict) -> float:
    score = 0

    # category
    if intent.get("category") and intent["category"] == candidate.get("category"):
        score += WEIGHTS["category"]

    # make
    if intent.get("make"):
        sim_brand = text_similarity(intent["make"], candidate.get("brand", ""))
        sim_compat = text_similarity(intent["make"], candidate.get("compatible_make", ""))
        score += WEIGHTS["make"] * max(sim_brand, sim_compat)
    
    # model
    if intent.get("model"):
        sim = text_similarity(intent["model"], candidate.get("compatible_model", ""))
        score += WEIGHTS["model"] * sim

    # year
    if intent.get("year") is not None:
        y = intent["year"]
        y_start = candidate.get("compatible_year_start")
        y_end = candidate.get("compatible_year_end")
        if y_start is not None and y_end is not None:
            if y_start <= y <= y_end:
                score += WEIGHTS["year"]
            else:
                distance = min(abs(y - y_start), abs(y - y_end))
                # if it's over 3 years 0, below 3 years - subtract with linearity
                partial = max(0.0, 1 - distance / 3)
                score += WEIGHTS["year"] * partial
   
    return score / MAX_SCORE

RESPONSE_PROMPT = """You are now recommending final output. You should give top 1 product with reason (based on the scoring)
And also give 2 other options. You should give reason why - not too long but just in one sentence.

User query: {query}

Ranked candidates (best first):
{ranked_text}
"""

def format_candidates(ranked: list) -> str:
    lines = []
    for i, c in enumerate(ranked, start=1):
        lines.append(
            f"{i}. {c.get('part_name')} - ${c.get('price_usd')} "
            f"(compatible: {c.get('compatible_make')} {c.get('compatible_model')}, "
            f"{c.get('compatible_year_start')}-{c.get('compatible_year_end')})"
        )
    return "\n".join(lines)


def response(state: AgentState) -> AgentState:
    """Second LLM call for the final response"""
    if not state["ranked"]:
        return {"response": "Sorry, unavailabe."}

    ranked_text = format_candidates(state["ranked"])
    prompt = RESPONSE_PROMPT.format(query=state["query"], ranked_text=ranked_text)
    llm_response = llm.invoke(prompt)
    return {"response" : llm_response.content}


# Build the Graph
builder = StateGraph(AgentState)
builder.add_node("understanding", query_understanding)
builder.add_node("clarify", clarify)
builder.add_node("retrieval", retrieval)
builder.add_node("ranking", ranking)
builder.add_node("response", response)

builder.set_entry_point("understanding")
builder.add_conditional_edges("understanding", route_after_understanding, {
    "clarify": "clarify",
    "retrieval": "retrieval",
})
builder.add_edge("clarify", END)
builder.add_edge("retrieval", "ranking")
builder.add_edge("ranking", "response")
builder.add_edge("response", END)

app = builder.compile()



if __name__ == "__main__":
    test_state = {"query": "Recommend me 2018 Honda Civic Brake Pad", "intent": {}, "is_ambiguous": False,
                  "candidates": [], "ranked": [], "response": ""}
    result = app.invoke(test_state)
    print(result["response"])
    