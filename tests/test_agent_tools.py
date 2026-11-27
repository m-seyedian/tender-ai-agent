from app.agent.tools import AGENT_TOOLS_SCHEMA

def test_all_five_agent_tools_have_valid_claude_schemas():
    expected = {'search_tenders', 'get_tender_details', 'extract_requirements', 'check_deadline', 'score_qualification'}
    assert {t['name'] for t in AGENT_TOOLS_SCHEMA} == expected
    for t in AGENT_TOOLS_SCHEMA:
        assert len(t['description']) >= 80
        assert t['input_schema']['type'] == 'object'
        assert set(t['input_schema']['required']).issubset(t['input_schema']['properties'].keys())
