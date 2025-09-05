import re


example_regex = re.compile("www.example.com")

def match_example(url):
    if example_regex.match(url):  # Check if example_regex matches the