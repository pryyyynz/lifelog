# Bugfix Requirements Document

## Introduction

The lifelog chatbot's query intent classification is broken in two related ways. First, the system returns hardcoded, prescriptive responses for conversational queries (e.g., "Hi" always yields the same greeting string, "thanks" always yields the same thanks string). This is undesirable — the user explicitly does not want prescribed per-pattern chatbot responses. Second, the intent classification is a binary regex-based split between "conversational" and "search", with no distinction between queries that reference prior conversation context (follow-ups) and queries that are genuine small talk (chit-chat). This means follow-up queries like "more from that session" or "what else happened then" can be misclassified, and chit-chat is handled with hardcoded strings rather than dynamic responses.

The fix replaces `is_conversational_query()` / `conversational_reply()` with a proper three-way `IntentClassifier` that classifies every query as `follow_up`, `retrieve`, or `chit_chat`, and routes chit-chat to a dynamic (LLM-generated or generic non-prescriptive fallback) response instead of a hardcoded string.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a user sends a greeting query (e.g., "Hi", "Hello", "Good morning") THEN the system returns a hardcoded greeting string that is always identical regardless of context

1.2 WHEN a user sends a thanks query (e.g., "thanks", "thank you") THEN the system returns a hardcoded thanks string that is always identical regardless of context

1.3 WHEN a user sends any other small-talk query matched by `_META_PATTERNS` or `_PLEASANTRY` THEN the system returns one of several hardcoded per-pattern strings from `conversational_reply()`

1.4 WHEN a user sends a follow-up query that references prior conversation context (e.g., "more from that session", "what else happened then") AND that query also matches a conversational pattern THEN the system may classify it as conversational and skip retrieval entirely, losing the follow-up intent

1.5 WHEN a user sends a chit-chat query THEN the system classifies it as either "conversational" or "search" with no third `chit_chat` category, making it impossible for downstream code to distinguish chit-chat from follow-ups

### Expected Behavior (Correct)

2.1 WHEN a user sends a greeting or small-talk query THEN the system SHALL respond with a dynamically generated reply (via LLM if available, or a single generic non-prescriptive fallback if not), NOT a hardcoded per-pattern string

2.2 WHEN a user sends a query that references prior conversation context (e.g., "more from that session", "what else happened then") THEN the system SHALL classify it as `follow_up` and route it to retrieval with context resolution, regardless of whether it also superficially resembles small talk

2.3 WHEN a user sends a query that should trigger retrieval from ingested data THEN the system SHALL classify it as `retrieve` and route it to the retrieval pipeline

2.4 WHEN a user sends a chit-chat query THEN the system SHALL classify it as `chit_chat` (a distinct third intent) and respond without performing retrieval

2.5 WHEN the system classifies a query as `chit_chat` and no LLM is configured THEN the system SHALL respond with a single generic non-prescriptive fallback message rather than a pattern-matched hardcoded string

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a user sends a query with explicit filters (source_type, session_id, date_from, date_to) THEN the system SHALL CONTINUE TO route it to retrieval regardless of query text

3.2 WHEN a user sends a retrieval query (e.g., "what did I do last summer?", "photos from the market") THEN the system SHALL CONTINUE TO classify it as `retrieve` and execute the full retrieval pipeline

3.3 WHEN a user sends a follow-up query that matches `_SESSION_PATTERNS` or `_TEMPORAL_PATTERNS` in `ConversationManager.resolve_context()` THEN the system SHALL CONTINUE TO resolve session and temporal references from prior conversation turns

3.4 WHEN the `ConversationManager` stores and retrieves conversation turns THEN the system SHALL CONTINUE TO persist turns, apply TTL expiry, and return `ResolvedContext` with the same fields and semantics

3.5 WHEN a query triggers a clarification prompt (ambiguous session reference) THEN the system SHALL CONTINUE TO return a `QueryResponse` with `clarification_prompt` set and empty `sessions`

3.6 WHEN a query is classified as `chit_chat` or `follow_up` THEN the system SHALL CONTINUE TO store the turn in `ConversationManager` so subsequent follow-up queries have context
