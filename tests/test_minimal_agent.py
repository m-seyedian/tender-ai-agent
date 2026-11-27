from app.agent.minimal_tool_demo import OfflineMockAnthropicClient, run_minimal_tool_loop

def test_minimal_claude_tool_use_loop():
    client = OfflineMockAnthropicClient()
    result = run_minimal_tool_loop('Quelle est la date limite du marché 26-104520 ?', client=client)
    assert len(result.tool_calls_executed) == 1
    assert result.tool_calls_executed[0]['name'] == 'get_tender_by_id'
    assert '26-104520' in result.final_answer
    assert result.total_input_tokens == 800
    assert result.total_output_tokens == 137
    assert result.estimated_cost_usd > 0
