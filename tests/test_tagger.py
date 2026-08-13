"""Tests for wallatag.tagger: the KeywordTagger and LLMTagger engines."""

import unittest

from wallatag.config import FocusGroup
from wallatag.llm import LLMError
from wallatag.tagger import KeywordTagger, LLMTagger, TagSuggestion


def make_tagger(
    groups,
    max_suggestions=10,
    tag_policy="prefer-existing",
    existing_tags=(),
):
    return KeywordTagger(
        groups,
        max_suggestions=max_suggestions,
        tag_policy=tag_policy,
        existing_tags=existing_tags,
    )


def entry(title="", url="", domain_name="", content=""):
    return {
        "id": 1,
        "title": title,
        "url": url,
        "domain_name": domain_name,
        "content": content,
        "reading_time": 1,
        "language": "en",
        "tags": [],
        "is_archived": 0,
        "is_starred": 0,
    }


class KeywordMatchingTest(unittest.TestCase):
    def test_matches_in_url_not_title(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        e = entry(title="focus timer", url="https://example.com/articles/pomodoro")
        result = make_tagger(groups).suggest(e)

        self.assertEqual([s.tag for s in result], ["productivity"])
        self.assertEqual(result[0].source, "rules")

    def test_matches_in_domain_name(self):
        groups = {"a": FocusGroup(keywords=("rust",), tags=("programming",))}
        e = entry(title="x", domain_name="blog.rust-lang.org")
        result = make_tagger(groups).suggest(e)

        self.assertEqual([s.tag for s in result], ["programming"])

    def test_matches_in_content_field(self):
        groups = {"a": FocusGroup(keywords=("recipe",), tags=("cooking",))}
        e = entry(title="article", content="<p>a tasty recipe for soup</p>")
        result = make_tagger(groups).suggest(e)

        self.assertEqual([s.tag for s in result], ["cooking"])

    def test_matching_is_case_insensitive(self):
        groups = {"a": FocusGroup(keywords=("GtD",), tags=("method",))}
        e = entry(title="my gtd setup")
        result = make_tagger(groups).suggest(e)

        self.assertEqual([s.tag for s in result], ["method"])

    def test_no_keyword_match_contributes_no_tags(self):
        groups = {"a": FocusGroup(keywords=("gtd",), tags=("method",))}
        e = entry(title="a recipe for soup", content="nothing relevant here")
        self.assertEqual(make_tagger(groups).suggest(e), [])

    def test_groups_and_tags_in_config_order(self):
        groups = {
            "first": FocusGroup(keywords=("alpha",), tags=("a1", "a2")),
            "second": FocusGroup(keywords=("beta",), tags=("b1",)),
        }
        result = make_tagger(groups, tag_policy="all").suggest(
            entry(title="alpha beta")
        )

        self.assertEqual([s.tag for s in result], ["a1", "a2", "b1"])


class PerFieldMatchingTest(unittest.TestCase):
    def test_blank_keyword_never_fires_group(self):
        groups = {
            "blank": FocusGroup(keywords=("",), tags=("t1",)),
            "spaces": FocusGroup(keywords=("   ",), tags=("t2",)),
        }
        e = entry(title="anything at all")
        result = make_tagger(groups, tag_policy="all").suggest(e)

        self.assertEqual(result, [])

    def test_multi_word_keyword_does_not_match_across_fields(self):
        # "rust programming" spans the title/url boundary and must NOT match.
        groups = {"a": FocusGroup(keywords=("rust programming",), tags=("t1",))}
        e = entry(title="the rust", url="programming-guide")
        result = make_tagger(groups).suggest(e)

        self.assertEqual(result, [])

    def test_multi_word_keyword_matches_within_single_field(self):
        groups = {"a": FocusGroup(keywords=("rust programming",), tags=("t1",))}
        e = entry(title="the rust programming guide")
        result = make_tagger(groups).suggest(e)

        self.assertEqual([s.tag for s in result], ["t1"])

    def test_multi_word_label_does_not_match_across_fields(self):
        result = make_tagger({}, existing_tags=["rust programming"]).suggest(
            entry(title="the rust", url="programming-guide")
        )

        self.assertEqual(result, [])

    def test_missing_fields_do_not_crash(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        # No domain_name key, content is None; both keyword and label match
        # within the remaining title field.
        e = {"id": 1, "title": "pomodoro notes", "content": None}
        result = make_tagger(groups, existing_tags=["notes"]).suggest(e)

        self.assertEqual([s.tag for s in result], ["notes", "productivity"])


class VocabularyTest(unittest.TestCase):
    def test_vocabulary_suggestion_preserves_original_label(self):
        result = make_tagger({}, existing_tags=["Pomodoro", "unrelated"]).suggest(
            entry(title="I use the pomodoro method")
        )

        self.assertEqual(
            result,
            [TagSuggestion(tag="Pomodoro", source="vocabulary", confidence=1.0)],
        )

    def test_vocabulary_order_follows_input_order(self):
        result = make_tagger({}, existing_tags=["beta", "alpha"]).suggest(
            entry(title="alpha beta")
        )

        self.assertEqual([s.tag for s in result], ["beta", "alpha"])

    def test_overlapping_labels_not_deduped(self):
        # "python" and "python3" are distinct casefolded labels -> both kept.
        result = make_tagger({}, existing_tags=["python", "python3"]).suggest(
            entry(title="python3 tutorial")
        )

        self.assertEqual([s.tag for s in result], ["python", "python3"])

    def test_non_string_existing_tag_skipped(self):
        result = make_tagger({}, existing_tags=[123, "python"]).suggest(
            entry(title="python tutorial")
        )

        self.assertEqual([s.tag for s in result], ["python"])

    def test_whitespace_padded_label_still_matches(self):
        result = make_tagger({}, existing_tags=["  python  "]).suggest(
            entry(title="python tutorial")
        )

        self.assertEqual([s.tag for s in result], ["  python  "])


class TagPolicyTest(unittest.TestCase):
    def test_only_existing_drops_rule_tags(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        tagger = make_tagger(
            groups, tag_policy="only-existing", existing_tags=["Pomodoro"]
        )
        result = tagger.suggest(entry(title="pomodoro focus"))

        self.assertEqual([s.tag for s in result], ["Pomodoro"])
        self.assertEqual([s.source for s in result], ["vocabulary"])

    def test_prefer_existing_orders_vocabulary_first(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        tagger = make_tagger(
            groups, tag_policy="prefer-existing", existing_tags=["Pomodoro"]
        )
        result = tagger.suggest(entry(title="pomodoro focus"))

        self.assertEqual([s.tag for s in result], ["Pomodoro", "productivity"])
        self.assertEqual([s.source for s in result], ["vocabulary", "rules"])

    def test_all_keeps_both_in_same_order(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        e = entry(title="pomodoro focus")
        all_result = make_tagger(
            groups, tag_policy="all", existing_tags=["Pomodoro"]
        ).suggest(e)
        prefer_result = make_tagger(
            groups, tag_policy="prefer-existing", existing_tags=["Pomodoro"]
        ).suggest(e)

        # "all" keeps rule-derived tags alongside the vocabulary match...
        self.assertEqual([s.tag for s in all_result], ["Pomodoro", "productivity"])
        # ...and is documented to be identical to "prefer-existing" in the MVP.
        self.assertEqual(all_result, prefer_result)

    def test_rule_only_when_no_vocabulary(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        tagger = make_tagger(groups, tag_policy="only-existing")
        result = tagger.suggest(entry(title="pomodoro"))

        self.assertEqual(result, [])

    def test_dedup_prefers_vocabulary_over_rules(self):
        # The tag is both an existing vocabulary label and a rule-derived tag:
        # it appears once, as the vocabulary suggestion.
        groups = {
            "a": FocusGroup(
                keywords=("pomodoro",), tags=("Pomodoro", "productivity")
            )
        }
        tagger = make_tagger(groups, tag_policy="all", existing_tags=["Pomodoro"])
        result = tagger.suggest(entry(title="pomodoro focus"))

        self.assertEqual([s.tag for s in result], ["Pomodoro", "productivity"])
        self.assertEqual(result[0].source, "vocabulary")
        self.assertEqual(result[0].confidence, 1.0)


class MaxSuggestionsTest(unittest.TestCase):
    def test_max_suggestions_caps_total(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("t1", "t2", "t3"))}
        result = make_tagger(groups, max_suggestions=2, tag_policy="all").suggest(
            entry(title="pomodoro")
        )

        self.assertEqual([s.tag for s in result], ["t1", "t2"])

    def test_max_suggestions_zero_yields_empty(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("t1",))}
        result = make_tagger(groups, max_suggestions=0).suggest(
            entry(title="pomodoro")
        )

        self.assertEqual(result, [])


class DeterminismTest(unittest.TestCase):
    def test_same_input_same_output(self):
        groups = {
            "a": FocusGroup(keywords=("pomodoro",), tags=("t1",)),
            "b": FocusGroup(keywords=("gtd",), tags=("t2",)),
        }
        tagger = make_tagger(
            groups, tag_policy="all", existing_tags=["Ztag"]
        )
        e = entry(title="pomodoro gtd ztag")

        self.assertEqual(tagger.suggest(e), tagger.suggest(e))
        self.assertEqual(
            [s.tag for s in tagger.suggest(e)], ["Ztag", "t1", "t2"]
        )


class ValidationTest(unittest.TestCase):
    def test_invalid_tag_policy_raises(self):
        with self.assertRaises(ValueError):
            make_tagger({}, tag_policy="nonsense")

    def test_negative_max_suggestions_raises(self):
        with self.assertRaises(ValueError):
            make_tagger({}, max_suggestions=-1)

    def test_bool_max_suggestions_raises(self):
        # bool is an int subclass; a non-negative int check must reject it.
        with self.assertRaises(ValueError):
            make_tagger({}, max_suggestions=True)


class KeywordTaggerEnableSwitchTest(unittest.TestCase):
    """enable_vocabulary / enable_rules off-switches gate the two sources."""

    def test_enable_vocabulary_false_drops_vocabulary_keeps_rules(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        tagger = KeywordTagger(
            groups,
            max_suggestions=10,
            tag_policy="prefer-existing",
            existing_tags=["Pomodoro"],
            enable_vocabulary=False,
        )
        result = tagger.suggest(entry(title="pomodoro focus"))
        # Only the rule-derived tag survives; the vocabulary label is gone.
        self.assertEqual([s.tag for s in result], ["productivity"])
        self.assertEqual([s.source for s in result], ["rules"])

    def test_enable_rules_false_drops_rules_keeps_vocabulary(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        tagger = KeywordTagger(
            groups,
            max_suggestions=10,
            tag_policy="prefer-existing",
            existing_tags=["Pomodoro"],
            enable_rules=False,
        )
        result = tagger.suggest(entry(title="pomodoro focus"))
        self.assertEqual([s.tag for s in result], ["Pomodoro"])
        self.assertEqual([s.source for s in result], ["vocabulary"])

    def test_both_false_yields_empty(self):
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        tagger = KeywordTagger(
            groups,
            max_suggestions=10,
            tag_policy="prefer-existing",
            existing_tags=["Pomodoro"],
            enable_vocabulary=False,
            enable_rules=False,
        )
        self.assertEqual(tagger.suggest(entry(title="pomodoro focus")), [])

    def test_enable_rules_false_also_wins_over_only_existing(self):
        # enable_rules=False drops rules even under tag_policy="all"; combined
        # with "only-existing" both gates agree and the result is vocabulary
        # only (rules are dropped by the policy AND the switch).
        groups = {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))}
        tagger = KeywordTagger(
            groups,
            max_suggestions=10,
            tag_policy="only-existing",
            existing_tags=["Pomodoro"],
            enable_rules=False,
        )
        result = tagger.suggest(entry(title="pomodoro focus"))
        self.assertEqual([s.tag for s in result], ["Pomodoro"])
        self.assertEqual([s.source for s in result], ["vocabulary"])

    def test_defaults_keep_both_sources(self):
        # No kwargs -> existing behavior (both sources fire).
        tagger = KeywordTagger(
            {"a": FocusGroup(keywords=("pomodoro",), tags=("productivity",))},
            max_suggestions=10,
            tag_policy="prefer-existing",
            existing_tags=["Pomodoro"],
        )
        self.assertTrue(tagger.enable_vocabulary)
        self.assertTrue(tagger.enable_rules)
        result = tagger.suggest(entry(title="pomodoro focus"))
        self.assertEqual([s.tag for s in result], ["Pomodoro", "productivity"])
        self.assertEqual([s.source for s in result], ["vocabulary", "rules"])

    def test_non_bool_enable_vocabulary_raises(self):
        with self.assertRaises(ValueError) as ctx:
            KeywordTagger(
                {},
                max_suggestions=10,
                tag_policy="prefer-existing",
                enable_vocabulary=1,
            )
        self.assertIn("enable_vocabulary", str(ctx.exception))

    def test_non_bool_enable_rules_raises(self):
        with self.assertRaises(ValueError) as ctx:
            KeywordTagger(
                {},
                max_suggestions=10,
                tag_policy="prefer-existing",
                enable_rules="no",
            )
        self.assertIn("enable_rules", str(ctx.exception))


class FakeLLMClient:
    """Stub LLMClientLike: returns canned JSON and records the prompts."""

    def __init__(self, body):
        self.body = body
        self.system_prompts = []
        self.user_prompts = []
        self.calls = 0

    def complete(self, system_prompt, user_prompt):
        self.calls += 1
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        return self.body


class RaisingLLMClient:
    """Stub whose complete() always raises LLMError (model down)."""

    def complete(self, system_prompt, user_prompt):
        raise LLMError("model unavailable", status=500)


def make_llm_tagger(
    client,
    groups=None,
    max_suggestions=10,
    tag_policy="prefer-existing",
    existing_tags=(),
    confidence_threshold=0.7,
    use_focus_groups=True,
):
    return LLMTagger(
        client,
        focus_groups=groups or {},
        max_suggestions=max_suggestions,
        tag_policy=tag_policy,
        existing_tags=existing_tags,
        confidence_threshold=confidence_threshold,
        use_focus_groups=use_focus_groups,
    )


class LLMTaggerSuggestTest(unittest.TestCase):
    """suggest(): parsing, threshold, policy, dedup, cap, error propagation."""

    def test_happy_path_parses_tags_and_confidences(self):
        client = FakeLLMClient(
            '[{"tag": "python", "confidence": 0.9}, '
            '{"tag": "rust", "confidence": 0.8}]'
        )
        result = make_llm_tagger(client).suggest(entry(title="python and rust"))

        self.assertEqual([s.tag for s in result], ["python", "rust"])
        self.assertEqual([s.source for s in result], ["llm", "llm"])
        self.assertEqual([s.confidence for s in result], [0.9, 0.8])

    def test_below_threshold_filtered_out(self):
        client = FakeLLMClient(
            '[{"tag": "low", "confidence": 0.5}, '
            '{"tag": "high", "confidence": 0.9}]'
        )
        result = make_llm_tagger(client).suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["high"])

    def test_confidence_clamped_to_unit_range(self):
        # White-box: _parse_response clamps confidence into [0, 1]. A clamped
        # 0.0 can never survive suggest() (any valid threshold is > 0), so the
        # clamp itself is asserted here, not via the filtered output.
        tagger = make_llm_tagger(FakeLLMClient("[]"))
        parsed = tagger._parse_response(
            '[{"tag": "too-high", "confidence": 1.5}, '
            '{"tag": "negative", "confidence": -0.5}]'
        )
        self.assertEqual(
            [(item["tag"], item["confidence"]) for item in parsed],
            [("too-high", 1.0), ("negative", 0.0)],
        )

    def test_overconfident_tag_survives_threshold_at_clamped_value(self):
        client = FakeLLMClient('[{"tag": "too-high", "confidence": 1.5}]')
        result = make_llm_tagger(client).suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["too-high"])
        self.assertEqual(result[0].confidence, 1.0)

    def test_only_existing_drops_non_vocabulary_tags(self):
        client = FakeLLMClient(
            '[{"tag": "python", "confidence": 0.9}, '
            '{"tag": "invented", "confidence": 0.9}]'
        )
        tagger = make_llm_tagger(
            client, tag_policy="only-existing", existing_tags=["python"]
        )
        result = tagger.suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["python"])

    def test_prefer_existing_keeps_all_above_threshold(self):
        client = FakeLLMClient(
            '[{"tag": "python", "confidence": 0.9}, '
            '{"tag": "newone", "confidence": 0.8}]'
        )
        tagger = make_llm_tagger(
            client, tag_policy="prefer-existing", existing_tags=["python"]
        )
        result = tagger.suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["python", "newone"])

    def test_all_policy_keeps_all_above_threshold(self):
        client = FakeLLMClient(
            '[{"tag": "python", "confidence": 0.9}, '
            '{"tag": "newone", "confidence": 0.8}]'
        )
        tagger = make_llm_tagger(client, tag_policy="all", existing_tags=["python"])
        result = tagger.suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["python", "newone"])

    def test_only_existing_case_insensitive_vocabulary(self):
        client = FakeLLMClient(
            '[{"tag": "Python", "confidence": 0.9}, '
            '{"tag": "rust", "confidence": 0.9}]'
        )
        tagger = make_llm_tagger(
            client, tag_policy="only-existing", existing_tags=["python"]
        )
        result = tagger.suggest(entry(title="x"))

        # Casefolded "python" is in the vocabulary, "rust" is not.
        self.assertEqual([s.tag for s in result], ["Python"])

    def test_case_insensitive_dedup_keeps_first(self):
        client = FakeLLMClient(
            '[{"tag": "Python", "confidence": 0.9}, '
            '{"tag": "python", "confidence": 0.8}]'
        )
        result = make_llm_tagger(client).suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["Python"])
        self.assertEqual(result[0].confidence, 0.9)

    def test_max_suggestions_caps_output(self):
        client = FakeLLMClient(
            '[{"tag": "a", "confidence": 0.9}, '
            '{"tag": "b", "confidence": 0.9}, '
            '{"tag": "c", "confidence": 0.9}]'
        )
        result = make_llm_tagger(client, max_suggestions=2).suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["a", "b"])

    def test_model_output_order_kept(self):
        client = FakeLLMClient(
            '[{"tag": "zeta", "confidence": 0.9}, '
            '{"tag": "alpha", "confidence": 0.9}]'
        )
        result = make_llm_tagger(client).suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["zeta", "alpha"])

    def test_empty_result_yields_no_suggestions(self):
        result = make_llm_tagger(FakeLLMClient("[]")).suggest(entry(title="x"))
        self.assertEqual(result, [])

    def test_llm_error_propagates(self):
        tagger = make_llm_tagger(RaisingLLMClient())
        with self.assertRaises(LLMError):
            tagger.suggest(entry(title="x"))

    def test_llm_error_carries_status(self):
        with self.assertRaises(LLMError) as ctx:
            make_llm_tagger(RaisingLLMClient()).suggest(entry(title="x"))
        self.assertEqual(ctx.exception.status, 500)

    def test_whitespace_padded_tag_stripped(self):
        # The model returned "  python  "; the suggestion must carry the
        # normalized tag so wallabag never receives padded labels.
        client = FakeLLMClient('[{"tag": "  python  ", "confidence": 0.9}]')
        result = make_llm_tagger(client).suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["python"])

    def test_threshold_boundary_keeps_equal_drops_below(self):
        # confidence == threshold is KEPT (>=), one epsilon below is dropped.
        client = FakeLLMClient(
            '[{"tag": "exact", "confidence": 0.7}, '
            '{"tag": "below", "confidence": 0.6999999}]'
        )
        result = make_llm_tagger(client, confidence_threshold=0.7).suggest(
            entry(title="x")
        )

        self.assertEqual([s.tag for s in result], ["exact"])

    def test_only_existing_tolerates_non_string_vocabulary(self):
        # A non-string element in existing_tags must not crash the
        # only-existing gate (it is filtered defensively) and the gate still
        # applies to the string vocabulary.
        client = FakeLLMClient(
            '[{"tag": "Python", "confidence": 0.9}, '
            '{"tag": "invented", "confidence": 0.9}]'
        )
        tagger = make_llm_tagger(
            client,
            tag_policy="only-existing",
            existing_tags=[123, "python"],
        )
        result = tagger.suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["Python"])


class LLMTaggerMalformedResponseTest(unittest.TestCase):
    def test_invalid_json_raises_llm_error(self):
        client = FakeLLMClient("not json at all")
        with self.assertRaises(LLMError):
            make_llm_tagger(client).suggest(entry(title="x"))

    def test_non_array_json_raises_llm_error(self):
        client = FakeLLMClient('{"tag": "solo", "confidence": 0.9}')
        with self.assertRaises(LLMError) as ctx:
            make_llm_tagger(client).suggest(entry(title="x"))
        self.assertIn("not a JSON array", str(ctx.exception))

    def test_entries_with_missing_fields_skipped(self):
        client = FakeLLMClient(
            '[{"confidence": 0.9}, '          # missing tag
            '{"tag": 123, "confidence": 0.8},'  # non-string tag
            '{"tag": "ok", "confidence": 0.7},'  # valid
            '{"tag": "badconf", "confidence": "high"},'  # non-numeric confidence
            '{"tag": "   ", "confidence": 0.9},'  # blank tag
            '{"tag": "boolconf", "confidence": true}]'  # bool confidence
        )
        result = make_llm_tagger(client).suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["ok"])

    def test_non_dict_entries_skipped(self):
        client = FakeLLMClient(
            '[42, "python", {"tag": "ok", "confidence": 0.8}]'
        )
        result = make_llm_tagger(client).suggest(entry(title="x"))

        self.assertEqual([s.tag for s in result], ["ok"])


class LLMTaggerPromptTest(unittest.TestCase):
    """The prompts sent to the client must carry vocabulary, focus, policy."""

    def test_system_prompt_contains_vocabulary_rule_verbatim(self):
        client = FakeLLMClient("[]")
        make_llm_tagger(client).suggest(entry(title="x"))

        prompt = client.system_prompts[0]
        self.assertIn(
            "prefer the same tags that already exist; only add new ones if they "
            "really don't fit and are an important part of the text",
            prompt,
        )

    def test_system_prompt_contains_focus_areas_and_vocabulary(self):
        groups = {
            "langs": FocusGroup(keywords=("python",), tags=("programming",)),
            "methods": FocusGroup(keywords=("gtd",), tags=("method",)),
        }
        client = FakeLLMClient("[]")
        make_llm_tagger(
            client, groups=groups, existing_tags=["rust", "python"]
        ).suggest(entry(title="x"))

        prompt = client.system_prompts[0]
        self.assertIn("programming", prompt)
        self.assertIn("method", prompt)
        self.assertIn("rust", prompt)
        self.assertIn("python", prompt)

    def test_system_prompt_mentions_none_when_empty(self):
        client = FakeLLMClient("[]")
        make_llm_tagger(client).suggest(entry(title="x"))

        prompt = client.system_prompts[0]
        self.assertIn("Existing tag vocabulary: none", prompt)
        self.assertIn("Focus areas: none", prompt)

    def test_system_prompt_only_existing_instructs_vocabulary_only(self):
        client = FakeLLMClient("[]")
        make_llm_tagger(
            client, tag_policy="only-existing", existing_tags=["python"]
        ).suggest(entry(title="x"))

        prompt = client.system_prompts[0]
        self.assertIn("ONLY choose from the provided existing tag vocabulary", prompt)

    def test_user_prompt_contains_cleaned_content_and_metadata(self):
        client = FakeLLMClient("[]")
        e = entry(
            title="A title",
            url="https://example.com/article",
            domain_name="example.com",
            content="<p>hello <b>world</b></p>",
        )
        make_llm_tagger(client).suggest(e)

        prompt = client.user_prompts[0]
        self.assertIn("Title: A title", prompt)
        self.assertIn("URL: https://example.com/article", prompt)
        self.assertIn("Domain: example.com", prompt)
        self.assertIn("Language: en", prompt)
        self.assertIn("Reading time: 1 minutes", prompt)
        # HTML stripped and whitespace collapsed.
        self.assertIn("Content: hello world", prompt)

    def test_user_prompt_content_truncated(self):
        client = FakeLLMClient("[]")
        e = entry(title="t", content="<p>" + "word " * 2000 + "</p>")
        make_llm_tagger(client).suggest(e)

        prompt = client.user_prompts[0]
        # 2000 words * ~5 chars > 6000-char cap -> truncated.
        self.assertLess(len(prompt), 7000)


class LLMTaggerFocusAreasTest(unittest.TestCase):
    """use_focus_groups and fields=() control the "Focus areas" prompt line."""

    def _prompt(self, **kwargs):
        client = FakeLLMClient("[]")
        make_llm_tagger(client, **kwargs).suggest(entry(title="x"))
        return client.system_prompts[0]

    def test_use_focus_groups_false_omits_focus_areas_line(self):
        # False -> the system prompt has NO "Focus areas" line at all (not
        # even "Focus areas: none").
        groups = {"langs": FocusGroup(keywords=("python",), tags=("programming",))}
        prompt = self._prompt(groups=groups, use_focus_groups=False)
        self.assertNotIn("Focus areas", prompt)
        # The rest of the prompt survives.
        self.assertIn("Existing tag vocabulary: none", prompt)
        self.assertIn("Return at most 10 tags.", prompt)

    def test_disabled_group_excluded_from_focus_areas(self):
        # fields=() group is excluded from the LLM focus areas; a normal
        # group's tags are still present.
        groups = {
            "disabled": FocusGroup(
                keywords=("pomodoro",), tags=("productivity",), fields=()
            ),
            "enabled": FocusGroup(keywords=("python",), tags=("programming",)),
        }
        prompt = self._prompt(groups=groups)
        self.assertIn("Focus areas: programming.", prompt)
        self.assertIn("programming", prompt)
        self.assertNotIn("productivity", prompt)

    def test_all_groups_disabled_yields_focus_areas_none(self):
        # use_focus_groups=True (default) with EVERY focus group having
        # fields == () -> no group contributes tags, so the prompt degrades
        # to "Focus areas: none." Combined with use_focus_groups=False, where
        # the same all-disabled groups drop the "Focus areas" line entirely.
        groups = {
            "a": FocusGroup(
                keywords=("python",), tags=("programming",), fields=()
            ),
            "b": FocusGroup(
                keywords=("pomodoro",), tags=("productivity",), fields=()
            ),
        }
        prompt = self._prompt(groups=groups)
        self.assertIn("Focus areas: none.", prompt)
        self.assertNotIn("programming", prompt)
        self.assertNotIn("productivity", prompt)
        prompt = self._prompt(groups=groups, use_focus_groups=False)
        self.assertNotIn("Focus areas", prompt)

    def test_fields_none_tags_kept_in_focus_areas(self):
        # Existing behavior pinned: fields=None groups contribute their tags.
        groups = {
            "default": FocusGroup(keywords=("python",), tags=("programming",)),
        }
        prompt = self._prompt(groups=groups)
        self.assertIn("Focus areas: programming.", prompt)

    def test_default_use_focus_groups_present(self):
        # Backward compat: constructing without use_focus_groups keeps the
        # "Focus areas" line in the prompt.
        client = FakeLLMClient("[]")
        LLMTagger(
            client,
            focus_groups={"a": FocusGroup(keywords=("k",), tags=("t",))},
            max_suggestions=5,
            tag_policy="prefer-existing",
        ).suggest(entry(title="x"))
        self.assertIn("Focus areas", client.system_prompts[0])

    def test_non_bool_use_focus_groups_raises(self):
        with self.assertRaises(ValueError) as ctx:
            make_llm_tagger(FakeLLMClient("[]"), use_focus_groups=1)
        self.assertIn("use_focus_groups", str(ctx.exception))


class LLMTaggerValidationTest(unittest.TestCase):
    def test_invalid_tag_policy_raises(self):
        with self.assertRaises(ValueError):
            make_llm_tagger(FakeLLMClient("[]"), tag_policy="nonsense")

    def test_negative_max_suggestions_raises(self):
        with self.assertRaises(ValueError):
            make_llm_tagger(FakeLLMClient("[]"), max_suggestions=-1)

    def test_bool_max_suggestions_raises(self):
        with self.assertRaises(ValueError):
            make_llm_tagger(FakeLLMClient("[]"), max_suggestions=True)

    def test_bad_confidence_thresholds_raise(self):
        for bad in (0, 1.5, -0.1, True):
            with self.assertRaises(ValueError):
                make_llm_tagger(FakeLLMClient("[]"), confidence_threshold=bad)

    def test_boundary_thresholds_accepted(self):
        client = FakeLLMClient('[{"tag": "t", "confidence": 1.0}]')
        result = make_llm_tagger(client, confidence_threshold=1.0).suggest(
            entry(title="x")
        )
        self.assertEqual([s.tag for s in result], ["t"])


class PerSourceMatchFieldsTest(unittest.TestCase):
    """vocabulary_fields and FocusGroup.fields gate which fields are matched.

    The vocabulary matcher checks only ``vocabulary_fields``; each rule group
    checks only its own ``fields`` (None -> all four, () -> never matches).
    """

    def test_vocabulary_fields_restrict_vocabulary_matching(self):
        tagger = KeywordTagger(
            {},
            max_suggestions=10,
            tag_policy="prefer-existing",
            existing_tags=["pomodoro"],
            vocabulary_fields=("title",),
        )
        # The label appears only in content: title-only matching -> no match.
        self.assertEqual(
            tagger.suggest(entry(title="focus", content="pomodoro notes")), []
        )
        # The label appears in the title: matches.
        result = tagger.suggest(entry(title="pomodoro focus", content="x"))
        self.assertEqual([s.tag for s in result], ["pomodoro"])
        self.assertEqual([s.source for s in result], ["vocabulary"])

    def test_vocabulary_fields_default_is_all_four(self):
        tagger = make_tagger({}, existing_tags=["rust"])
        self.assertEqual(
            tagger.vocabulary_fields, ("title", "url", "domain_name", "content")
        )
        # A label only in the URL still matches with the default fields.
        result = tagger.suggest(entry(title="x", url="https://blog.rust-lang.org"))
        self.assertEqual([s.tag for s in result], ["rust"])

    def test_group_fields_restrict_rule_matching(self):
        groups = {
            "content_only": FocusGroup(
                keywords=("recipe",), tags=("cooking",), fields=("content",)
            )
        }
        tagger = make_tagger(groups)
        # Keyword in the title only: content-only matching -> no match.
        self.assertEqual(tagger.suggest(entry(title="recipe", content="")), [])
        # Keyword in the content field: matches.
        result = tagger.suggest(entry(title="x", content="a tasty recipe"))
        self.assertEqual([s.tag for s in result], ["cooking"])

    def test_group_fields_none_matches_all_four(self):
        groups = {
            "url_only_default": FocusGroup(
                keywords=("pomodoro",), tags=("productivity",)
            )
        }
        # fields defaults to None -> the keyword matches in the URL field.
        result = make_tagger(groups).suggest(
            entry(title="x", url="https://example.com/pomodoro")
        )
        self.assertEqual([s.tag for s in result], ["productivity"])

    def test_group_empty_fields_matches_nothing(self):
        groups = {
            "disabled": FocusGroup(
                keywords=("pomodoro",), tags=("productivity",), fields=()
            )
        }
        tagger = make_tagger(groups)
        # The keyword is in every default field, yet the empty tuple disables
        # the group entirely.
        self.assertEqual(
            tagger.suggest(
                entry(title="pomodoro", url="pomodoro", content="pomodoro")
            ),
            [],
        )

    def test_per_group_fields_independent(self):
        # Group A matches title only; group B matches content only. Each
        # keyword fires on its own field subset.
        groups = {
            "titles": FocusGroup(keywords=("gtd",), tags=("method",), fields=("title",)),
            "bodies": FocusGroup(
                keywords=("recipe",), tags=("cooking",), fields=("content",)
            ),
        }
        e = entry(title="gtd", content="a recipe")
        result = make_tagger(groups).suggest(e)
        self.assertEqual([s.tag for s in result], ["method", "cooking"])

    def test_invalid_vocabulary_fields_member_raises(self):
        with self.assertRaises(ValueError) as ctx:
            KeywordTagger(
                {},
                max_suggestions=1,
                tag_policy="all",
                vocabulary_fields=("author",),
            )
        message = str(ctx.exception)
        self.assertIn("author", message)
        self.assertIn("valid choices", message)
        self.assertIn("title", message)
        # Case-sensitive: "Title" is invalid too.
        with self.assertRaises(ValueError):
            KeywordTagger(
                {}, max_suggestions=1, tag_policy="all", vocabulary_fields=("Title",)
            )

    def test_field_needles_respects_given_fields(self):
        tagger = make_tagger({})
        e = entry(title="TITLE", url="URL", domain_name="DOMAIN", content="BODY")
        self.assertEqual(tagger._field_needles(e, ("title",)), ("title",))
        self.assertEqual(tagger._field_needles(e, ("url", "content")), ("url", "body"))
        # Non-string/missing values are skipped per field, as before.
        self.assertEqual(
            tagger._field_needles({"title": "t", "content": None}, ("title", "content")),
            ("t",),
        )

    def test_vocabulary_fields_empty_disables_vocabulary(self):
        tagger = KeywordTagger(
            {},
            max_suggestions=10,
            tag_policy="prefer-existing",
            existing_tags=["pomodoro"],
            vocabulary_fields=(),
        )
        self.assertEqual(tagger.suggest(entry(title="pomodoro")), [])

    def test_vocabulary_fields_one_shot_generator_materialized_once(self):
        # vocabulary_fields is only declared Iterable: a one-shot generator
        # must be materialized once and used for BOTH validation and matching.
        # Before the fix the validation loop consumed the generator, so the
        # stored tuple came out empty and vocabulary was silently disabled.
        tagger = KeywordTagger(
            {},
            max_suggestions=10,
            tag_policy="prefer-existing",
            existing_tags=["pomodoro"],
            vocabulary_fields=(x for x in ["title"]),
        )
        self.assertEqual(tagger.vocabulary_fields, ("title",))
        # Matching still works: the title-only label is found in the title.
        result = tagger.suggest(entry(title="pomodoro focus", content="x"))
        self.assertEqual([s.tag for s in result], ["pomodoro"])
        self.assertEqual([s.source for s in result], ["vocabulary"])

    def test_vocabulary_fields_empty_generator_disables_vocabulary(self):
        # An exhausted/empty generator materializes to () -> vocabulary off,
        # exactly like an explicitly empty tuple.
        tagger = KeywordTagger(
            {},
            max_suggestions=10,
            tag_policy="prefer-existing",
            existing_tags=["pomodoro"],
            vocabulary_fields=(x for x in []),
        )
        self.assertEqual(tagger.vocabulary_fields, ())
        self.assertEqual(tagger.suggest(entry(title="pomodoro")), [])


if __name__ == "__main__":
    unittest.main()
