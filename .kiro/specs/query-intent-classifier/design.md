# Query Intent Classifier Bugfix Design

## Overview

The current `chat_intent.py` module uses a binary regex-based split between "conversational" and "search" queries, returning hardcoded per-pattern strings for chit-chat. This design replaces `is_conversational_query()` / `conversational_reply()` with a three-way `IntentClassifier` that classifies every query as `follow_up`, `retrieve`, or `chit_chat`.

The fix is targeted and minimal:
- `app/retrieval/chat_intent.py` is replaced with a new `IntentClassifier` class
- `app/api/main.py` is updated to use the new classifier and route each intent appropriately
- `app/retrieval/conversation.py` and `app/retrieval/query_analyzer.py` are **not modified**

Chit-chat responses become dynamic: LLM-generated when an LLM client is available, or a single generic non-prescriptive fallback string when it is not. No per-pattern hardcoded strings remain.

---

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug — a query is chit-chat but the system returns a hardcoded per-pattern string, OR a follow-up query is misclassified as chit-chat and skips retrieval
- **Property (P)**: The desired behavior — chit-chat gets a dynamic reply; follow-ups route to retrieval; retrieval queries execute the full pipeline
- **Preservation**: All existing `ConversationManager`, retrieval pipeline, clarification prompt, and turn-storage behavior that must remain unchanged
- **IntentClassifier**: The new class in `app/retrieval/chat_intent.py` that replaces `is_conversational_query()` and `conversational_reply()`
- **QueryIntent**: An enum with three values: `follow_up`, `retrieve`, `chit_chat`
- **isBugCondition**: Pseudocode predicate identifying inputs that trigger the defect
- **conversational_reply()**: The old function in `chat_intent.py` that returned hardcoded per-pattern strings — to be deleted
- **is_conversational_query()**: The old function in `chat_intent.py` that performed binary classification — to be deleted
- **has_filters**: Boolean flag indicating whether the request carries explicit API filters (source_type, session_id, date_from, date_to)
- **LLM client**: An optional language model client (e.g., OpenAI, Ollama) used to generate dynamic chit-chat replies
- **generic_fallback**: A single non-prescriptive fallback string returned when no LLM is configured

---

## Bug Details

### Bug Condition

The bug manifests in two related ways:

1. **Hardcoded chit-chat replies**: When a query is classified as conversational, `conversational_reply()` pattern-matches the query text and returns one of ~8 distinct hardcoded strings. The reply is always identical for the same pattern, regardless of context or conversation history.

2. **Follow-up misclassification**: A query like "more from that session" or "what else happened then" may match both `_SESSION_PATTERNS` (in `ConversationManager`) and `_META_PATTERNS` / `_PLEASANTRY` (in `is_conversational_query()`). The binary classifier has no follow-up category, so it can route a follow-up to the conversational path and skip retrieval entirely.

**Formal Specification:**

```
FUNCTION isBugCondition(query, has_filters)
  INPUT: query: str, has_filters: bool
  OUTPUT: boolean

  -- Arm 1: hardcoded reply bug
  IF is_conversational_query(query, has_filters=has_filters) THEN
    reply := conversational_reply(query)
    RETURN reply IN HARDCODED_REPLY_SET
  END IF

  -- Arm 2: follow-up misclassification bug
  IF matches_follow_up_pattern(query)
     AND is_conversational_query(query, has_filters=False) THEN
    RETURN True   -- follow-up incorrectly skips retrieval
  END IF

  RETURN False
END FUNCTION
```

Where `HARDCODED_REPLY_SET` is the finite set of strings that `conversational_reply()` can return (the ~8 hardcoded reply strings in the current implementation).

### Examples

- **"Hi"** → current: hardcoded greeting string; expected: dynamic LLM reply or single generic fallback
- **"thanks"** → current: hardcoded thanks string; expected: dynamic LLM reply or single generic fallback
- **"more from that session"** → current: may match `_PLEASANTRY` or fall through to `_META_PATTERNS` and skip retrieval; expected: classified as `follow_up`, routed to retrieval with context resolution
- **"what else happened then"** → current: matches `_META_PATTERNS` (`\bwhat do you do\b` does not match, but the binary classifier has no follow-up path); expected: classified as `follow_up`
- **"what did I do last summer?"** → current: correctly routes to retrieval via `_SEARCH_OVERRIDE_PATTERNS`; expected: unchanged, classified as `retrieve`

---

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Queries with explicit API filters (`source_type`, `session_id`, `date_from`, `date_to`) must always route to retrieval, regardless of query text
- Retrieval queries (matching `_SEARCH_OVERRIDE_PATTERNS` or containing temporal signals) must continue to execute the full retrieval pipeline
- `ConversationManager.resolve_context()` must be called before classification, and its output (`effective_query`, `session_id_filter`, `temporal_range_override`, `clarification_needed`) must be used exactly as before
- `ConversationManager.store_turn()` must be called after every response — including `chit_chat` and `follow_up` responses — so subsequent queries have context
- Clarification prompt behavior: when `resolve_context()` returns `clarification_needed=True`, the endpoint must return `QueryResponse` with `clarification_prompt` set and empty `sessions`
- All `ConversationManager` internals (TTL expiry, persistence, `_SESSION_PATTERNS`, `_TEMPORAL_PATTERNS`) must remain unchanged

**Scope:**
All inputs that do NOT trigger the bug condition should be completely unaffected by this fix. This includes:
- Any query with `has_filters=True`
- Any query matching `_SEARCH_OVERRIDE_PATTERNS`
- Any query with a temporal signal detected by `_extract_temporal()`
- All `ConversationManager` operations

---

## Hypothesized Root Cause

Based on the bug description and code analysis:

1. **No follow-up intent category**: `is_conversational_query()` returns a boolean. There is no third state for follow-up queries. The `ConversationManager` pattern matching (`_SESSION_PATTERNS`, `_TEMPORAL_PATTERNS`) happens in `resolve_context()`, but the conversational check in `post_query` runs after `resolve_context()` and can still intercept follow-up queries if they superficially match `_META_PATTERNS` or `_PLEASANTRY`.

2. **Hardcoded reply dispatch**: `conversational_reply()` uses a chain of `re.search()` calls to select from ~8 fixed strings. There is no LLM call, no randomness, and no context awareness. The same input always produces the same output.

3. **Pattern ordering in `post_query`**: The `is_conversational_query()` check in `post_query` runs after `resolve_context()` but before `_query_analyzer.analyze()`. A follow-up query that resolves context correctly can still be short-circuited by the conversational check if it matches a pleasantry pattern.

4. **No LLM integration point**: `chat_intent.py` has no mechanism to call an LLM. Adding dynamic replies requires introducing an optional LLM client dependency.

---

## Correctness Properties

Property 1: Bug Condition — Chit-Chat Gets Dynamic Reply

_For any_ query where `isBugCondition` holds (the query is chit-chat and the old code would return a hardcoded per-pattern string), the fixed `IntentClassifier` SHALL classify the query as `chit_chat` and return a reply that is NOT a member of `HARDCODED_REPLY_SET`. When no LLM is configured, the reply SHALL be the single generic fallback string. When an LLM is configured, the reply SHALL be the LLM-generated string.

**Validates: Requirements 2.1, 2.4, 2.5**

Property 2: Bug Condition — Follow-Up Queries Route to Retrieval

_For any_ query that matches `_SESSION_PATTERNS` or `_TEMPORAL_PATTERNS` (as defined in `conversation.py`), the fixed `IntentClassifier` SHALL classify the query as `follow_up` and the `post_query` endpoint SHALL route it to the retrieval pipeline with context resolution applied, regardless of whether the query also superficially matches chit-chat patterns.

**Validates: Requirements 2.2**

Property 3: Preservation — Filters Always Route to Retrieval

_For any_ query where `has_filters=True`, the fixed `IntentClassifier` SHALL classify the query as `retrieve`, preserving the existing behavior that explicit API filters always bypass intent classification.

**Validates: Requirements 3.1**

Property 4: Preservation — ConversationManager Behavior Unchanged

_For any_ sequence of `store_turn()` and `resolve_context()` calls, the fixed code SHALL produce the same `ResolvedContext` outputs and persistence behavior as the original code, preserving all TTL, session-reference, and temporal-reference resolution semantics.

**Validates: Requirements 3.3, 3.4**

---

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `app/retrieval/chat_intent.py`

**Changes**:
1. **Delete** `is_conversational_query()` and `conversational_reply()`
2. **Add** `QueryIntent` enum with values `follow_up`, `retrieve`, `chit_chat`
3. **Add** `IntentClassifier` class with:
   - `__init__(self, llm_client=None)` — accepts optional LLM client
   - `classify(self, query: str, *, has_filters: bool = False) -> QueryIntent` — three-way classification
   - `chit_chat_reply(self, query: str) -> str` — dynamic reply via LLM or generic fallback
4. **Classification logic** in `classify()`:
   - If `has_filters` → return `retrieve`
   - If query matches `_SESSION_PATTERNS` or `_TEMPORAL_PATTERNS` → return `follow_up`
   - If `_extract_temporal(query)` is not None → return `retrieve`
   - If query matches `_SEARCH_OVERRIDE_PATTERNS` → return `retrieve`
   - If query matches `_PLEASANTRY` or `_META_PATTERNS` → return `chit_chat`
   - Default → return `retrieve`
5. **Reply logic** in `chit_chat_reply()`:
   - If LLM client available → call LLM with a brief system prompt and return generated text
   - Otherwise → return `GENERIC_FALLBACK` (a single non-prescriptive string, e.g. `"I'm here to help you search your life log. What would you like to find?"`)

**File**: `app/api/main.py`

**Changes**:
1. **Replace** the `from app.retrieval.chat_intent import conversational_reply, is_conversational_query` import with `from app.retrieval.chat_intent import IntentClassifier, QueryIntent`
2. **Instantiate** `_intent_classifier = IntentClassifier(llm_client=None)` alongside other singletons (or in lifespan, wiring in an LLM client from config if available)
3. **Replace** the `if is_conversational_query(...)` block with:
   ```python
   intent = _intent_classifier.classify(request.query, has_filters=has_filters)
   if intent == QueryIntent.chit_chat:
       reply = _intent_classifier.chit_chat_reply(request.query)
       _conv_manager.store_turn(...)
       return QueryResponse(sessions=[], conversation_id=conv_id,
                            chat_message=reply, query_debug={"intent": "chit_chat"})
   # follow_up and retrieve both fall through to the existing retrieval pipeline
   ```
4. **Update** `query_debug["intent"]` to use `intent.value` (i.e., `"follow_up"`, `"retrieve"`, or `"chit_chat"`)

**File**: `app/retrieval/conversation.py` — **no changes**

**File**: `app/retrieval/query_analyzer.py` — **no changes**

---

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that call `is_conversational_query()` and `conversational_reply()` directly on known inputs, and assert that the replies are NOT hardcoded strings. Also test that follow-up queries are not misclassified. Run these tests on the UNFIXED code to observe failures and understand the root cause.

**Test Cases**:
1. **Hardcoded greeting test**: Call `conversational_reply("Hi")` and assert the result is not the known hardcoded greeting string — will fail on unfixed code
2. **Hardcoded thanks test**: Call `conversational_reply("thanks")` and assert the result is not the known hardcoded thanks string — will fail on unfixed code
3. **Follow-up misclassification test**: Call `is_conversational_query("more from that session")` and assert it returns `False` — will fail on unfixed code if the query matches pleasantry/meta patterns
4. **Follow-up with temporal ref**: Call `is_conversational_query("what else happened then")` and assert it returns `False` — may fail on unfixed code

**Expected Counterexamples**:
- `conversational_reply("Hi")` returns the exact hardcoded string `"Hi! I'm your Life Log assistant..."`
- `conversational_reply("thanks")` returns the exact hardcoded string `"You're welcome! Ask me anytime..."`
- `is_conversational_query("more from that session")` may return `True` depending on pattern matching order

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed `IntentClassifier` produces the expected behavior.

**Pseudocode:**
```
FOR ALL query WHERE isBugCondition(query, has_filters=False) DO
  intent := IntentClassifier.classify(query, has_filters=False)
  IF query matches follow_up patterns THEN
    ASSERT intent == QueryIntent.follow_up
  ELSE
    ASSERT intent == QueryIntent.chit_chat
    reply := IntentClassifier.chit_chat_reply(query)
    ASSERT reply NOT IN HARDCODED_REPLY_SET
    IF no LLM configured THEN
      ASSERT reply == GENERIC_FALLBACK
    END IF
  END IF
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed code produces the same result as the original code.

**Pseudocode:**
```
FOR ALL query WHERE NOT isBugCondition(query, has_filters) DO
  intent_fixed := IntentClassifier.classify(query, has_filters=has_filters)
  -- Retrieval queries must still classify as retrieve
  IF is_retrieval_query(query) OR has_filters THEN
    ASSERT intent_fixed == QueryIntent.retrieve
  END IF
  -- ConversationManager must produce identical outputs
  ctx_original := ConversationManager.resolve_context(query, conv_id)
  ctx_fixed    := ConversationManager.resolve_context(query, conv_id)
  ASSERT ctx_original == ctx_fixed
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for retrieval queries and filter-bearing queries, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Filter preservation**: For any query with `has_filters=True`, verify `classify()` returns `retrieve`
2. **Retrieval query preservation**: For queries matching `_SEARCH_OVERRIDE_PATTERNS`, verify `classify()` returns `retrieve`
3. **Temporal query preservation**: For queries with a temporal signal, verify `classify()` returns `retrieve`
4. **ConversationManager preservation**: Verify `resolve_context()` and `store_turn()` produce identical outputs before and after the fix

### Unit Tests

- Test `IntentClassifier.classify()` for each of the three intents with representative inputs
- Test `IntentClassifier.chit_chat_reply()` with no LLM configured — must return `GENERIC_FALLBACK`
- Test `IntentClassifier.chit_chat_reply()` with a mock LLM client — must return the mock's output
- Test that `has_filters=True` always returns `retrieve` regardless of query text
- Test follow-up pattern queries return `follow_up`
- Test edge cases: empty query, query matching both follow-up and chit-chat patterns

### Property-Based Tests

- Generate random chit-chat queries (from the known pattern set) and verify the reply is never a member of `HARDCODED_REPLY_SET`
- Generate random queries with `has_filters=True` and verify `classify()` always returns `retrieve`
- Generate random retrieval-like queries and verify `classify()` returns `retrieve`
- Generate random follow-up queries (from `_SESSION_PATTERNS` / `_TEMPORAL_PATTERNS`) and verify `classify()` returns `follow_up`

### Integration Tests

- Test the full `POST /query` flow with a chit-chat query — verify `chat_message` is set, `sessions` is empty, and `store_turn` was called
- Test the full `POST /query` flow with a follow-up query — verify retrieval pipeline executes and context is resolved
- Test the full `POST /query` flow with a retrieval query — verify sessions are returned and `query_debug.intent` is `"retrieve"`
- Test that switching between chit-chat and retrieval queries in the same conversation preserves context across turns
