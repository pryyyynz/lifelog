# Implementation Plan

## Overview

This task list implements the three-way `IntentClassifier` bugfix for the query intent classifier. It follows the exploratory bugfix workflow: write exploration tests first (on unfixed code), write preservation tests (on unfixed code), then implement the fix and verify both test suites pass.

## Task Dependency Graph

```json
{
  "waves": [
    {"wave": 1, "tasks": ["1", "2"]},
    {"wave": 2, "tasks": ["3.1"]},
    {"wave": 3, "tasks": ["3.2", "3.3"]},
    {"wave": 4, "tasks": ["3.4"]},
    {"wave": 5, "tasks": ["3.5", "3.6"]},
    {"wave": 6, "tasks": ["4"]}
  ]
}
```

## Tasks

- [ ] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Hardcoded Chit-Chat Replies and Follow-Up Misclassification
  - **CRITICAL**: This test MUST FAIL on unfixed code — failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior — it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to the concrete failing cases from the bug condition
  - Write tests in `tests/test_query_intent_classifier_bugfix.py`
  - Test 1 — Hardcoded greeting: call `conversational_reply("Hi")` and assert the result is NOT the known hardcoded string `"Hi! I'm your Life Log assistant. I search your personal history — try \"what did I do last summer?\" or \"photos from the market.\""` — will FAIL on unfixed code
  - Test 2 — Hardcoded thanks: call `conversational_reply("thanks")` and assert the result is NOT the known hardcoded string `"You're welcome! Ask me anytime you want to search your journals, photos, videos, or other indexed memories."` — will FAIL on unfixed code
  - Test 3 — Follow-up misclassification: call `is_conversational_query("more from that session")` and assert it returns `False` — will FAIL on unfixed code if the query matches pleasantry/meta patterns
  - Test 4 — Temporal follow-up misclassification: call `is_conversational_query("what else happened then")` and assert it returns `False` — will FAIL on unfixed code
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests FAIL (this is correct — it proves the bug exists)
  - Document counterexamples found (e.g., `conversational_reply("Hi")` returns the exact hardcoded greeting string; `is_conversational_query("more from that session")` returns `True`)
  - Mark task complete when tests are written, run, and failures are documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Retrieval Queries and Filter-Bearing Queries Always Route to Retrieval
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs (queries where `isBugCondition` returns False)
  - Observe: `is_conversational_query("what did I do last summer?", has_filters=False)` returns `False` on unfixed code
  - Observe: `is_conversational_query("photos from the market", has_filters=False)` returns `False` on unfixed code
  - Observe: `is_conversational_query("hello", has_filters=True)` returns `False` on unfixed code (filters override)
  - Observe: `is_conversational_query("show me videos from June")` returns `False` on unfixed code
  - Write property-based tests in `tests/test_query_intent_classifier_bugfix.py` using `hypothesis`:
    - **Property 2a**: For any query with `has_filters=True`, `is_conversational_query(query, has_filters=True)` returns `False` — generate queries from `_META_PATTERNS` and `_PLEASANTRY` inputs with `has_filters=True`
    - **Property 2b**: For any query matching `_SEARCH_OVERRIDE_PATTERNS` (e.g., queries starting with "what did I", "show me", "find"), `is_conversational_query(query)` returns `False`
    - **Property 2c**: For any query containing a temporal signal (e.g., "last summer", "yesterday", "in 2023"), `is_conversational_query(query)` returns `False`
  - Verify all property tests PASS on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2_

- [ ] 3. Fix: Replace binary classifier with three-way IntentClassifier

  - [ ] 3.1 Replace `app/retrieval/chat_intent.py` with `QueryIntent` enum and `IntentClassifier` class
    - Delete `is_conversational_query()` and `conversational_reply()` functions
    - Add `QueryIntent` enum with three values: `follow_up`, `retrieve`, `chit_chat`
    - Add `GENERIC_FALLBACK` constant: a single non-prescriptive string (e.g., `"I'm here to help you search your life log. What would you like to find?"`)
    - Add `HARDCODED_REPLY_SET` constant: the finite set of strings that `conversational_reply()` could return (used in tests to assert the fix eliminates them)
    - Add `IntentClassifier` class with `__init__(self, llm_client=None)`
    - Implement `classify(self, query: str, *, has_filters: bool = False) -> QueryIntent`:
      - If `has_filters` → return `QueryIntent.retrieve`
      - If query matches any pattern in `_SESSION_PATTERNS` or `_TEMPORAL_PATTERNS` (from `conversation.py`) → return `QueryIntent.follow_up`
      - If `_extract_temporal(query.lower())` is not None → return `QueryIntent.retrieve`
      - If query matches any pattern in `_SEARCH_OVERRIDE_PATTERNS` → return `QueryIntent.retrieve`
      - If query matches `_PLEASANTRY` or any pattern in `_META_PATTERNS` → return `QueryIntent.chit_chat`
      - Default → return `QueryIntent.retrieve`
    - Implement `chit_chat_reply(self, query: str) -> str`:
      - If `self._llm_client` is not None → call LLM with a brief system prompt and return generated text
      - Otherwise → return `GENERIC_FALLBACK`
    - _Bug_Condition: `isBugCondition(query, has_filters)` — query is chit-chat and `conversational_reply()` returns a member of `HARDCODED_REPLY_SET`, OR query matches follow-up patterns and `is_conversational_query()` returns `True`_
    - _Expected_Behavior: `IntentClassifier.classify()` returns `chit_chat` for chit-chat queries and `follow_up` for follow-up queries; `chit_chat_reply()` returns `GENERIC_FALLBACK` (no LLM) or LLM-generated text — never a member of `HARDCODED_REPLY_SET`_
    - _Preservation: Queries with `has_filters=True` → `retrieve`; queries matching `_SEARCH_OVERRIDE_PATTERNS` or temporal signals → `retrieve`; `ConversationManager` unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2_

  - [ ] 3.2 Update `app/api/main.py` to use `IntentClassifier` with three-way routing
    - Replace `from app.retrieval.chat_intent import conversational_reply, is_conversational_query` with `from app.retrieval.chat_intent import IntentClassifier, QueryIntent`
    - Add `_intent_classifier: IntentClassifier | None = None` to the module-level singletons block
    - Instantiate `_intent_classifier = IntentClassifier(llm_client=None)` in the `_lifespan` context manager alongside other singletons
    - In `post_query`, replace the `if is_conversational_query(...)` block with three-way routing:
      ```python
      intent = _intent_classifier.classify(request.query, has_filters=has_filters)
      if intent == QueryIntent.chit_chat:
          reply = _intent_classifier.chit_chat_reply(request.query)
          _conv_manager.store_turn(conv_id=conv_id, query=request.query,
                                   temporal_range=None, session_ids=[],
                                   place_names=[], result_count=0)
          return QueryResponse(sessions=[], conversation_id=conv_id,
                               chat_message=reply, query_debug={"intent": "chit_chat"})
      # follow_up and retrieve both fall through to the existing retrieval pipeline
      ```
    - Update `query_debug["intent"]` at the end of `post_query` to use `intent.value` (i.e., `"follow_up"` or `"retrieve"`)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.6_

  - [ ] 3.3 Write unit tests for `IntentClassifier` in `tests/test_query_intent_classifier_bugfix.py`
    - Test `classify()` returns `chit_chat` for greeting queries: `"Hi"`, `"hello"`, `"good morning"`
    - Test `classify()` returns `chit_chat` for thanks queries: `"thanks"`, `"thank you"`
    - Test `classify()` returns `chit_chat` for meta queries: `"what do you do?"`, `"who are you"`, `"help"`
    - Test `classify()` returns `follow_up` for session-reference queries: `"more from that session"`, `"that session"`, `"same session"`
    - Test `classify()` returns `follow_up` for temporal-reference queries: `"what else happened then"`, `"more from then"`, `"around that time"`
    - Test `classify()` returns `retrieve` for retrieval queries: `"what did I do last summer?"`, `"photos from the market"`, `"show me videos from June"`
    - Test `classify()` returns `retrieve` when `has_filters=True` regardless of query text (including chit-chat queries)
    - Test `chit_chat_reply()` with no LLM configured returns `GENERIC_FALLBACK`
    - Test `chit_chat_reply()` with a mock LLM client returns the mock's output (not `GENERIC_FALLBACK`)
    - Test edge case: empty query string → `classify()` returns `retrieve` (default)
    - Test edge case: query matching both follow-up and chit-chat patterns → `classify()` returns `follow_up` (follow-up takes priority)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 3.4 Write integration tests for the `POST /query` endpoint in `tests/test_query_intent_classifier_bugfix.py`
    - Test chit-chat query flow: POST `{"query": "Hi", "conversation_id": null}` → verify `chat_message` is set, `sessions` is empty, `query_debug.intent == "chit_chat"`, and `chat_message` is NOT a member of `HARDCODED_REPLY_SET`
    - Test follow-up query flow: POST `{"query": "more from that session", "conversation_id": <id>}` after a prior retrieval turn → verify retrieval pipeline executes and `query_debug.intent == "follow_up"`
    - Test retrieval query flow: POST `{"query": "what did I do last summer?"}` → verify `query_debug.intent == "retrieve"`
    - Test context persistence: send chit-chat query then follow-up query in same conversation → verify `store_turn` was called after chit-chat so the follow-up has context
    - Test filter override: POST `{"query": "Hi", "filters": {"source_type": "photos"}}` → verify `query_debug.intent == "retrieve"` (filters bypass intent classification)
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.6_

  - [ ] 3.5 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Chit-Chat Replies Are Dynamic, Follow-Ups Route to Retrieval
    - **IMPORTANT**: Re-run the SAME tests from task 1 — do NOT write new tests
    - The tests from task 1 encode the expected behavior
    - When these tests pass, it confirms the expected behavior is satisfied
    - Re-run `tests/test_query_intent_classifier_bugfix.py` exploration tests from step 1
    - **EXPECTED OUTCOME**: Tests PASS (confirms bug is fixed — `IntentClassifier` no longer returns hardcoded strings and follow-up queries are no longer misclassified)
    - _Requirements: 2.1, 2.2, 2.4, 2.5_

  - [ ] 3.6 Verify preservation tests still pass
    - **Property 2: Preservation** - Retrieval and Filter Routing Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions — filter-bearing queries and retrieval queries still route correctly)
    - Confirm all property tests still pass after fix (no regressions)

- [ ] 4. Checkpoint — Ensure all tests pass
  - Run the full test suite: `pytest tests/test_query_intent_classifier_bugfix.py tests/test_chat_intent.py tests/test_section15_16.py -v`
  - Note: `tests/test_chat_intent.py` imports `is_conversational_query` and `conversational_reply` — update those imports to use `IntentClassifier` and `QueryIntent` so the existing tests continue to pass with the new API
  - Verify all tests pass; ask the user if questions arise

## Notes

- `hypothesis` is required for property-based tests in tasks 1 and 2. Install with `pip install hypothesis` if not already present.
- The existing `tests/test_chat_intent.py` imports `is_conversational_query` and `conversational_reply` directly. After the fix (task 3.1), those symbols no longer exist — update that file's imports as part of the checkpoint (task 4) to use `IntentClassifier` and `QueryIntent`.
- `conversation.py` and `query_analyzer.py` are NOT modified by this fix.
- The `_SESSION_PATTERNS` and `_TEMPORAL_PATTERNS` lists live in `conversation.py` — import them from there (or duplicate the patterns) in `IntentClassifier.classify()` to detect follow-up queries.
- `HARDCODED_REPLY_SET` should be defined in the test file (not in production code) as the finite set of strings that the old `conversational_reply()` could return, used to assert the fix eliminates them.
