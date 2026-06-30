import os
import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

# Load environment variables securely
load_dotenv()

# ── MODEL INITIALIZATION ──
# Note: Ensure your .env file has GROQ_API_KEY1 matching this variable
api_key = os.getenv("GROQ_API_KEY1") or os.getenv("GROQ_API_KEY")
if not api_key:
    print("⚠️ WARNING: Groq API Key not found in environment variables.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0 # 0 for factual accuracy and consistent reasoning
)

# ── DEFINING TOOLS ──
@tool
def get_current_date() -> str:
    """Returns today's date. Use before searching when the query
    involves 'latest', 'current', 'today', or 'this year'."""
    return datetime.date.today().isoformat()

search = DuckDuckGoSearchRun(
    description=(
        "Search the web for real-time information. Use for current news, "
        "recent events, live data, prices, or anything published after 2024. "
        "Do NOT use for background knowledge, history, or encyclopedic definitions."
    )
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2),
    description=(
        "Look up encyclopedic information. Use for historical facts, "
        "scientific concepts, notable people, organizations, and background context. "
        "Do NOT use for current events, prices, or real-time news."
    )
)

tools = [search, wiki, get_current_date]

# ── SYSTEM PROMPT & ARCHITECTURE ──
TODAY = datetime.date.today().isoformat()
SYSTEM_PROMPT = f"""You are AgentX, a highly intelligent and precise research assistant.
Today's date is {TODAY}.

Follow these strict rules:
1. Always state which tool provided each piece of information in your final answer.
2. Structure your final answer cleanly (e.g., Introduction → Key Facts → Conclusion).
3. If a question asks about current events (after 2024), strictly use DuckDuckGo.
4. If a question asks about history or facts, strictly use Wikipedia.
5. If you cannot find accurate information after searching, admit it honestly. Do not hallucinate or guess.
"""

# ── AGENT ──
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt=SYSTEM_PROMPT, # 🌟 FINAL FIX: Changed to 'prompt'
)


# ── INFERENCE WITH REASONING TRACE ──
def run_agent_with_trace(user_input: str, session_id: str) -> tuple[str, str]:
    """Runs the LangGraph agent and captures both the final answer and the exact tools used."""
    trace_log = []
    final_answer = ""
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 10, # Prevents infinite loops if the agent gets stuck
    }

    try:
        # Stream the agent's thought process step-by-step
        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="values",
        ):
            last_message = event["messages"][-1]
            
            # Log Tool Calls for the UI Trace
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tc in last_message.tool_calls:
                    trace_log.append(f"🛠️ **Tool Activated:** `{tc['name']}`\n   *Query:* {tc['args']}")
            
            # Capture the final AI response (when no more tools are being called)
            elif last_message.type == "ai" and not getattr(last_message, "tool_calls", None):
                if last_message.content: 
                    final_answer = last_message.content

    except GraphRecursionError:
        final_answer = "⚠️ **System Limit Reached:** I couldn't find a definitive answer within the step limit. Please try narrowing down your question."
    except Exception as e:
        final_answer = f"❌ **Error encountered:** {str(e)}"

    trace_str = "\n\n".join(trace_log) if trace_log else "🧠 No external tools were required to answer this."
    return final_answer, trace_str