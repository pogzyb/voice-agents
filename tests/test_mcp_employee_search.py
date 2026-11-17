import json
import unittest
from unittest.mock import patch

from mcp import function_app
from mcp.function_app import app, fake_names, search_employees


class EmployeeSearchTests(unittest.TestCase):
    def search(self, **arguments):
        return json.loads(search_employees(json.dumps({"arguments": arguments})))

    def test_tool_exposes_optional_name_and_title(self):
        function = next(f for f in app.get_functions() if f.get_function_name() == "search_employees")
        trigger = json.loads(function.get_function_json())["bindings"][0]
        properties = json.loads(trigger["toolProperties"])
        self.assertEqual({p["propertyName"] for p in properties}, {"name", "title"})
        self.assertTrue(all(p["propertyType"] == "string" and not p["isRequired"] for p in properties))

    def test_name_search_is_fuzzy_and_case_insensitive(self):
        matches = self.search(name="  ALICE SINCLAR  ")
        self.assertEqual(len(matches), 3)
        self.assertEqual(matches[0], fake_names[0])

    def test_title_search_ranks_matching_titles_first(self):
        matches = self.search(title="mortgage consultant")
        self.assertEqual(len(matches), 3)
        self.assertTrue(all(employee["title"] == "Mortgage Consultant" for employee in matches))

    def test_both_fields_contribute_to_ranking(self):
        employees = [
            {"name": "Alex Smith", "title": "Loan Officer"},
            {"name": "Alex Smith", "title": "Mortgage Advisor"},
            {"name": "Alex Jones", "title": "Mortgage Advisor"},
        ]
        with patch.object(function_app, "fake_names", employees):
            matches = self.search(name="Alex Smith", title="Mortgage Advisr")
        self.assertEqual(matches[0], employees[1])

    def test_missing_empty_and_invalid_arguments_are_rejected(self):
        for arguments in ({}, {"name": " ", "title": ""}, {"name": None}, {"title": 123}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.search(**arguments)


if __name__ == "__main__":
    unittest.main()
