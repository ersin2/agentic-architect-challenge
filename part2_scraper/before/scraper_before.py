"""RECONSTRUCTION - this is NOT PointStar's original script.

The challenge describes a script that "scrapes a website and summarises the
content, but fails on complex pages or long content", without supplying the
code. This file is a short, realistic reconstruction of such a script, kept
here only to show the defects. Each defect is marked "# DEFECT n" and is
explained, with its fix, in docs/PART2_DIAGNOSIS.md.

The fixed version is the rest of part2_scraper/ (fetch, extract, render,
chunking, summarize, guardrail). Do not use this file.

(Imports are inside the functions only so the test suite can import this
module without installing `requests` or `openai`.)
"""

import sys


def scrape(url):
    import requests

    # DEFECT 1: no timeout (a slow server hangs the script forever), no status check
    # (a 404 or error page gets summarised), no size limit (a huge page fills memory),
    # no content-type check (PDF or image bytes are parsed as HTML).
    html = requests.get(url).text
    return extract_text(html)


def extract_text(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    # DEFECT 2: get_text() keeps all visible text of the whole page: menus, cookie banners,
    # ads, sidebars, related links, footers. On a complex page much of the text is this
    # noise, and the model is asked to summarise it. (Recent BeautifulSoup versions skip
    # <script> and <style> text by default, so code is not the main problem; layout is.)
    # DEFECT 3: a JavaScript-rendered page has almost no text in its HTML, so this
    # returns a near-empty string and the summary is about nothing.
    return soup.get_text()


def summarize(text):
    from openai import OpenAI

    client = OpenAI()
    # DEFECT 4 - THE BOTTLENECK: the whole raw page goes into ONE prompt. Long or noisy
    # pages exceed the context window (request rejected) or get cut (the summary only
    # covers the start), and cost and latency grow with every byte of boilerplate.
    # DEFECT 5: the page text is pasted in as if it were instructions, so text on the
    # page such as "ignore previous instructions" is obeyed (prompt injection).
    prompt = f"Summarize the following web page in under 100 words:\n\n{text}"
    # DEFECT 6: no timeout, retry or error handling around the model call.
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    # DEFECT 7: "under 100 words" is only a request. Nothing measures the output,
    # so a long summary is returned as is.
    return response.choices[0].message.content


if __name__ == "__main__":
    print(summarize(scrape(sys.argv[1])))
