import json
import sys
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from app.core.config import settings

# Pricing per 1M tokens (USD) for claude-sonnet-4-5
SONNET_INPUT_COST_PER_1M = 3.00
SONNET_OUTPUT_COST_PER_1M = 15.00

MOCK_TENDERS_DB: dict[str, dict[str, Any]] = {
    "26-104520": {
        "tender_id": "26-104520",
        "title": "Maintenance et hébergement souverain SecNumCloud d'une plateforme SI",
        "buyer": "Ministère de l'Intérieur - SGAMI",
        "department": "75",
        "cpv": "72267100",
        "estimated_budget_eur": 450000,
        "deadline": "2026-12-15T12:00:00Z",
        "status": "À étudier",
    },
    "26-104889": {
        "tender_id": "26-104889",
        "title": "Fourniture de mobilier de bureau ergonomique",
        "buyer": "Mairie de Lyon",
        "department": "69",
        "cpv": "39130000",
        "estimated_budget_eur": 85000,
        "deadline": "2026-12-03T16:00:00Z",
        "status": "En attente",
    },
}


# 1. JSON Schema definition for Claude Tool Use
GET_TENDER_BY_ID_TOOL: dict[str, Any] = {
    "name": "get_tender_by_id",
    "description": (
        "Retrieve structured metadata for a specific French public tender (BOAMP) "
        "using its unique tender_id (e.g., '26-104520'). Returns buyer, CPV code, "
        "department, estimated budget in EUR, submission deadline, and triage status."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tender_id": {
                "type": "string",
                "description": "The unique BOAMP tender identifier, e.g., '26-104520'.",
            }
        },
        "required": ["tender_id"],
    },
}


# 2. Local Python function executed when Claude requests the tool
def get_tender_by_id(tender_id: str) -> dict[str, Any]:
    tender = MOCK_TENDERS_DB.get(tender_id)
    if not tender:
        return {"found": False, "tender_id": tender_id, "error": "Tender not found in database."}
    return {"found": True, "tender": tender}


@dataclass
class AgentRunResult:
    final_answer: str
    tool_calls_executed: list[dict[str, Any]]
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float


def calculate_token_cost(input_tokens: int, output_tokens: int) -> float:
    in_cost = (input_tokens / 1_000_000) * SONNET_INPUT_COST_PER_1M
    out_cost = (output_tokens / 1_000_000) * SONNET_OUTPUT_COST_PER_1M
    return round(in_cost + out_cost, 6)


def run_minimal_tool_loop(
    user_query: str,
    client: Any | None = None,
    model: str | None = None,
    max_turns: int = 5,
) -> AgentRunResult:
    """Executes the complete Claude Messages API tool_use -> tool_result loop."""
    if client is None:
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is missing in .env. Pass a mock client or set ANTHROPIC_API_KEY.")
        client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    target_model = model or settings.CLAUDE_MODEL
    system_prompt = (
        "You are a French public procurement (BOAMP) assistant. "
        "Always use the `get_tender_by_id` tool when asked about a specific tender ID. "
        "Never guess tender budgets or deadlines without calling the tool first."
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_query}]
    tool_calls_log: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0

    for _ in range(max_turns):
        response = client.messages.create(
            model=target_model,
            max_tokens=1024,
            temperature=0.0,
            system=system_prompt,
            tools=[GET_TENDER_BY_ID_TOOL],
            messages=messages,
        )

        total_in += getattr(response.usage, "input_tokens", 0)
        total_out += getattr(response.usage, "output_tokens", 0)

        # Append assistant turn (contains text and/or tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            final_text = "\n".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            return AgentRunResult(
                final_answer=final_text,
                tool_calls_executed=tool_calls_log,
                total_input_tokens=total_in,
                total_output_tokens=total_out,
                estimated_cost_usd=calculate_token_cost(total_in, total_out),
            )

        if response.stop_reason == "tool_use":
            tool_results_blocks: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    if block.name == "get_tender_by_id":
                        output = get_tender_by_id(**block.input)
                    else:
                        output = {"error": f"Unknown tool: {block.name}"}

                    tool_calls_log.append(
                        {"tool_use_id": block.id, "name": block.name, "input": block.input, "output": output}
                    )
                    tool_results_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(output, ensure_ascii=False),
                        }
                    )

            # Send tool_result back as a user message
            messages.append({"role": "user", "content": tool_results_blocks})

    raise RuntimeError(f"Agent exceeded max_turns={max_turns} without reaching end_turn.")


class _MockBlock:
    def __init__(self, block_type: str, **kwargs: Any):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockResponse:
    def __init__(self, stop_reason: str, content: list[Any], input_tokens: int, output_tokens: int):
        self.stop_reason = stop_reason
        self.content = content
        self.usage = _MockUsage(input_tokens, output_tokens)


class OfflineMockAnthropicClient:
    """Simulates a 2-turn Claude tool_use -> tool_result cycle offline."""

    def __init__(self):
        self.messages = self
        self._turn = 0

    def create(self, **kwargs: Any) -> _MockResponse:
        self._turn += 1
        if self._turn == 1:
            return _MockResponse(
                stop_reason="tool_use",
                content=[
                    _MockBlock("text", text="Je vais vérifier les informations du marché 26-104520."),
                    _MockBlock(
                        "tool_use",
                        id="toolu_demo_001",
                        name="get_tender_by_id",
                        input={"tender_id": "26-104520"},
                    ),
                ],
                input_tokens=320,
                output_tokens=65,
            )
        # Turn 2: inspect the tool_result sent back in messages[-1]
        last_user_msg = kwargs["messages"][-1]["content"][0]
        payload = json.loads(last_user_msg["content"])
        tender = payload["tender"]
        return _MockResponse(
            stop_reason="end_turn",
            content=[
                _MockBlock(
                    "text",
                    text=(
                        f"Le marché **{tender['tender_id']}** ({tender['title']}) est publié par "
                        f"**{tender['buyer']}** avec un budget estimé de **{tender['estimated_budget_eur']:,} €** "
                        f"et une date limite au **{tender['deadline']}**."
                    ),
                )
            ],
            input_tokens=480,
            output_tokens=72,
        )


if __name__ == "__main__":
    use_mock = "--mock" in sys.argv or not settings.ANTHROPIC_API_KEY
    demo_client = OfflineMockAnthropicClient() if use_mock else None
    mode_label = "OFFLINE MOCK" if use_mock else f"LIVE CLAUDE ({settings.CLAUDE_MODEL})"

    print(f"--- Running Minimal Tool Use Loop [{mode_label}] ---")
    res = run_minimal_tool_loop("Quelle est la date limite et le budget du marché 26-104520 ?", client=demo_client)
    print("Tool Calls:", json.dumps(res.tool_calls_executed, indent=2, ensure_ascii=False))
    print("Final Answer:", res.final_answer)
    print(
        f"Token Usage -> Input: {res.total_input_tokens} | Output: {res.total_output_tokens} "
        f"| Est. Cost: ${res.estimated_cost_usd}"
    )
