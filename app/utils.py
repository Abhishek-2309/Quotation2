def build_prompt_from_doc(doc_text: str):
    return f"""
You are given a Quotation document of security services where you have to extract the following details given in the JSON schema

Map all the fields in the below JSON schema
Return it as follows:
~~~json
{{
  "Company Name": "...",
  "Wage Type": "...",
  "Unit Price($/Hr)": "...",
  "Project": "...",
  "Year Quoted": "...",
  "Notes": "..."
}}
~~~

Note:
Classify Wage Type as either Prevailing/Non Prevailing. If not found, mention in notes.
Unit Price Logic:
Unit Price($/Hr) per Hour refers to the Wages of the Security Guard per hour. 
If multiple Categories of guards are mentioned, mention them as separate rows in a table. 
If same category of guard has multiple subcategories, mention them as subrows within main row.
Each conditional charge alongside the base rate is a column. Unit Price should be the total after all these are considered.
After getting the unit price table, write it as a json entry within the schema with keys representing columns and value representing corresponding entry.

Extract only from document contents. If relevant values are not found, leave empty.

Document:
{doc_text}
"""
