import os
from typing import List
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.google import Gemini
from agno.os import AgentOS

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===========================
# Load environment variables
# ===========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
POLICY_PATH = "./data/policies/policies.txt"
FAISS_INDEX_PATH = "./faiss_index"
TOP_K = int(os.getenv("POLICY_TOP_K", "1"))  # top-1 by default for small files

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment")
if not os.path.exists(POLICY_PATH):
    raise FileNotFoundError(f"Policy file not found: {POLICY_PATH}")

# ===========================
# Load policy text
# ===========================
with open(POLICY_PATH, "r", encoding="utf-8") as f:
    policy_text = f.read().strip()
if not policy_text:
    raise ValueError("Policy file is empty")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "],
)
chunks: List[str] = splitter.split_text(policy_text)
policy_docs: List[Document] = [Document(page_content=c) for c in chunks]

# ===========================
# Build / load FAISS vectorstore
# ===========================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists(FAISS_INDEX_PATH):
    print("🔹 Loading existing FAISS index from disk...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    print("🔹 Creating FAISS index from policy documents...")
    vectorstore = FAISS.from_documents(policy_docs, embedding_model)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved at {FAISS_INDEX_PATH}")


retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ===========================
# Retriever wrapper function
# ===========================
def retrieve_policies(query: str) -> str:
    if not query or not query.strip():
        return "No relevant policies found."
    results: List[Document] = retriever.invoke(query.strip())  # use invoke() to avoid deprecation
    if not results:
        return "No relevant policies found."
    print("🔎 Retrieved chunks:", [doc.page_content for doc in results])
    return "\n\n".join(doc.page_content for doc in results)

# ===========================
# Policy-checking agent
# ===========================
policy_agent = Agent(
    name="PolicyChecker",
    model=Gemini(
        id="models/gemini-1.5-flash",
        api_key=GOOGLE_API_KEY,
    ),
    instructions=[
        "You are a strict policy compliance checker.",
        "Your name is Policy Checker Rahul.",
        "Always ground answers in company policies.",
        "Use the 'retrieve_policies' tool to fetch relevant policies.",
        "If a request violates a policy, clearly explain why.",
        "Always cite the policy number when possible.",
    ],
    tools=[retrieve_policies],
    markdown=True,
)

# ===========================
# Normal helper agent
# ===========================
helper_agent = Agent(
    name="Helper",
    model=Gemini(
        id="models/gemini-1.5-flash",
        api_key=GOOGLE_API_KEY,
    ),
    instructions=["You are a friendly and knowledgeable AI assistant."],
    markdown=True,
)

# ===========================
# Define AgentOS
# ===========================
agent_os = AgentOS(
    os_id="my-multi-agent-os",
    description="AgentOS with Gemini helper and policy checker using LangChain RAG.",
    agents=[helper_agent, policy_agent],
)

# ===========================
# Expose FastAPI app
# ===========================
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="my_os:app", reload=True)
