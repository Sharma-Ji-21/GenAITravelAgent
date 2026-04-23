"""
Multi-Agent Travel Planner System
==================================
Built with LangChain + LangGraph

Agents:
  1. InputParserAgent     — Extracts structured travel details from user input
  2. DestinationAgent     — Researches destination highlights & tips
  3. ItineraryAgent       — Builds a day-by-day travel itinerary
  4. BudgetAgent          — Estimates costs and budget breakdown
  5. SummaryAgent         — Compiles everything into a final travel plan

Workflow:  Input → Destination → Itinerary → Budget → Summary
"""

import os
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import operator
import json

# ─────────────────────────────────────────────
# 1.  SHARED STATE
# ─────────────────────────────────────────────

class TravelState(TypedDict):
    """Shared state passed between every agent node."""
    raw_input:        str                          # Original user query
    parsed_details:   dict                         # Extracted structured details
    destination_info: str                          # Research from DestinationAgent
    itinerary:        str                          # Day-by-day plan
    budget_breakdown: str                          # Cost estimate
    final_plan:       str                          # Compiled output
    messages:         Annotated[list, operator.add] # Full message history


# ─────────────────────────────────────────────
# 2.  LLM  (swap model name as needed)
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.7) -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Export it before running:\n"
            "  export GROQ_API_KEY=sk-..."
        )
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature, api_key=api_key)


# ─────────────────────────────────────────────
# 3.  AGENT NODES
# ─────────────────────────────────────────────

def input_parser_agent(state: TravelState) -> TravelState:
    """
    Agent 1 – InputParserAgent
    Reads the raw user message and extracts structured travel details as JSON.
    """
    print("\n[Agent 1] InputParserAgent — parsing user input …")
    llm = get_llm(temperature=0.0)

    system_prompt = """You are a travel query parser. Extract structured travel details 
from the user's input and return ONLY valid JSON with these keys:
  destination (string), origin (string or null), duration_days (int),
  num_travelers (int), travel_style (string: budget|mid-range|luxury),
  interests (list of strings), special_requirements (list of strings)

If a field cannot be determined, use a sensible default. No markdown, no extra text."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["raw_input"])
    ])

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback — extract what we can
        parsed = {
            "destination": "Unknown",
            "origin": None,
            "duration_days": 7,
            "num_travelers": 2,
            "travel_style": "mid-range",
            "interests": ["sightseeing", "food"],
            "special_requirements": []
        }

    print(f"   Parsed: {json.dumps(parsed, indent=2)}")
    return {
        **state,
        "parsed_details": parsed,
        "messages": [HumanMessage(content=f"Parsed details: {json.dumps(parsed)}")]
    }


def destination_agent(state: TravelState) -> TravelState:
    """
    Agent 2 – DestinationAgent
    Researches the destination: top attractions, best time to visit, local tips.
    """
    print("\n[Agent 2] DestinationAgent — researching destination …")
    llm = get_llm(temperature=0.7)
    details = state["parsed_details"]

    system_prompt = f"""You are an expert travel researcher specialising in {details.get('destination')}.
Provide concise but rich destination information covering:
  • Top 5–7 must-see attractions with brief descriptions
  • Local cuisine highlights (3–5 dishes/spots)
  • Cultural etiquette & practical tips
  • Best neighbourhoods to stay in
  • Getting around (transport tips)

Tailor the response for {details.get('travel_style', 'mid-range')} travellers 
interested in: {', '.join(details.get('interests', []))}."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research travel information for {details.get('destination')} "
                             f"for a {details.get('duration_days')}-day trip.")
    ])

    print("   Destination research complete.")
    return {
        **state,
        "destination_info": response.content,
        "messages": [HumanMessage(content=f"Destination info ready for {details.get('destination')}")]
    }


def itinerary_agent(state: TravelState) -> TravelState:
    """
    Agent 3 – ItineraryAgent
    Creates a detailed day-by-day itinerary using destination research.
    """
    print("\n[Agent 3] ItineraryAgent — building itinerary …")
    llm = get_llm(temperature=0.8)
    details = state["parsed_details"]

    system_prompt = """You are an expert travel planner. Using the destination research provided,
create a detailed day-by-day itinerary. For each day include:
  - Morning, Afternoon, Evening activities
  - Specific restaurant/café recommendations for meals
  - Estimated travel times between locations
  - Any booking tips or advance reservations needed

Format each day clearly as: Day N: [Theme Title]"""

    prompt = f"""Create a {details.get('duration_days')}-day itinerary for {details.get('destination')}.
Travelers: {details.get('num_travelers')} | Style: {details.get('travel_style')}
Interests: {', '.join(details.get('interests', []))}
Special requirements: {', '.join(details.get('special_requirements', [])) or 'None'}

Destination research to use:
{state['destination_info']}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])

    print("   Itinerary created.")
    return {
        **state,
        "itinerary": response.content,
        "messages": [HumanMessage(content="Day-by-day itinerary complete.")]
    }


def budget_agent(state: TravelState) -> TravelState:
    """
    Agent 4 – BudgetAgent
    Estimates costs across categories and provides a total budget breakdown.
    """
    print("\n[Agent 4] BudgetAgent — estimating budget …")
    llm = get_llm(temperature=0.3)
    details = state["parsed_details"]

    system_prompt = """You are a travel budget specialist. Provide realistic cost estimates 
in USD with the following breakdown:
  • Accommodation (per night × number of nights)
  • Flights / Transport (round trip estimate)
  • Daily food & dining
  • Activities & entrance fees
  • Local transport (taxis, metro, etc.)
  • Shopping & miscellaneous (10–15% buffer)
  • TOTAL estimated cost per person
  • TOTAL estimated cost for the group

Give low / mid / high ranges where appropriate."""

    prompt = f"""Estimate travel budget for:
Destination: {details.get('destination')}
Duration: {details.get('duration_days')} days
Travelers: {details.get('num_travelers')}
Style: {details.get('travel_style')}
Itinerary context: {state['itinerary'][:500]}…"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])

    print("   Budget breakdown complete.")
    return {
        **state,
        "budget_breakdown": response.content,
        "messages": [HumanMessage(content="Budget estimation complete.")]
    }


def summary_agent(state: TravelState) -> TravelState:
    """
    Agent 5 – SummaryAgent
    Compiles all agent outputs into one polished, shareable travel plan.
    """
    print("\n[Agent 5] SummaryAgent — compiling final travel plan …")
    llm = get_llm(temperature=0.6)
    details = state["parsed_details"]

    system_prompt = """You are a travel editor creating a polished, comprehensive travel plan document.
Combine all provided information into a well-structured, engaging travel guide with:
  1. Trip Overview (title, destination, dates, travelers)
  2. Destination Highlights (key info from research)
  3. Day-by-Day Itinerary (formatted and readable)
  4. Budget Summary (clear totals and tips)
  5. Quick Reference Tips (packing, emergency contacts format, apps to download)
  6. Closing remarks

Use clear headings, emojis for readability, and an enthusiastic yet professional tone."""

    prompt = f"""Compile the final travel plan:

TRIP DETAILS: {json.dumps(details, indent=2)}

DESTINATION RESEARCH:
{state['destination_info']}

ITINERARY:
{state['itinerary']}

BUDGET:
{state['budget_breakdown']}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])

    print("\n   ✅ Final travel plan compiled!")
    return {
        **state,
        "final_plan": response.content,
        "messages": [HumanMessage(content="Final travel plan ready.")]
    }


# ─────────────────────────────────────────────
# 4.  BUILD THE LANGGRAPH WORKFLOW
# ─────────────────────────────────────────────

def build_travel_graph() -> StateGraph:
    """
    Constructs the LangGraph workflow with 5 sequential agent nodes.

    Graph topology:
      input_parser → destination → itinerary → budget → summary → END
    """
    graph = StateGraph(TravelState)

    # Register nodes
    graph.add_node("input_parser",  input_parser_agent)
    graph.add_node("destination",   destination_agent)
    graph.add_node("itinerary",     itinerary_agent)
    graph.add_node("budget",        budget_agent)
    graph.add_node("summary",       summary_agent)

    # Define edges (linear pipeline)
    graph.set_entry_point("input_parser")
    graph.add_edge("input_parser", "destination")
    graph.add_edge("destination",  "itinerary")
    graph.add_edge("itinerary",    "budget")
    graph.add_edge("budget",       "summary")
    graph.add_edge("summary",      END)

    return graph.compile()


# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🌍  Multi-Agent AI Travel Planner")
    print("      Powered by LangChain + LangGraph")
    print("=" * 60)
    print()

    # Dynamic user input
    print("Describe your dream trip! Include destination, duration,")
    print("number of travelers, interests, and any special requirements.")
    print()
    user_input = input("Your travel request: ").strip()

    if not user_input:
        print("No input provided. Using demo query …")
        user_input = (
            "I want to travel to Kyoto, Japan for 7 days with my partner. "
            "We love temples, food, and traditional culture. Mid-range budget."
        )

    print(f"\nProcessing: \"{user_input}\"")
    print("-" * 60)

    # Build and run the graph
    app = build_travel_graph()

    initial_state: TravelState = {
        "raw_input":        user_input,
        "parsed_details":   {},
        "destination_info": "",
        "itinerary":        "",
        "budget_breakdown": "",
        "final_plan":       "",
        "messages":         []
    }

    result = app.invoke(initial_state)

    # Output
    print("\n" + "=" * 60)
    print("  ✈️  YOUR COMPLETE TRAVEL PLAN")
    print("=" * 60)
    print(result["final_plan"])

    # Optionally save to file
    output_file = "travel_plan.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Travel Plan\n\n**Generated by Multi-Agent AI Travel Planner**\n\n")
        f.write(result["final_plan"])
    print(f"\n💾 Plan saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()