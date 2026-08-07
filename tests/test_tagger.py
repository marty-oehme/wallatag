"""Tests for wallatag.tagger: the KeywordTagger suggestion engine."""

import unittest

from wallatag.config import FocusGroup
from wallatag.tagger import KeywordTagger, TagSuggestion


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


if __name__ == "__main__":
    unittest.main()
