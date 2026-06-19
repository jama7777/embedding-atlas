import os
import urllib.request
import re
from html.parser import HTMLParser

class SimpleWikiParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_content = []
        self.active_ignored_tags = set()
        self.in_body = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Start recording content when we hit the main article content
        if tag == "div" and attrs_dict.get("id") == "bodyContent":
            self.in_body = True
            
        if tag in {"script", "style", "noscript", "nav", "footer", "header", "table"}:
            self.active_ignored_tags.add(tag)

    def handle_endtag(self, tag):
        if tag in self.active_ignored_tags:
            self.active_ignored_tags.remove(tag)

    def handle_data(self, data):
        # Only record if we are inside bodyContent and not inside any ignored tags
        if self.in_body and not self.active_ignored_tags:
            self.text_content.append(data)

def main():
    url = "https://en.wikipedia.org/wiki/LIGO"
    print(f"Fetching content from: {url}...")
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            
        print(f"Fetched {len(html)} characters of HTML.")
        print("Parsing HTML content...")
        parser = SimpleWikiParser()
        parser.feed(html)
        
        raw_text = "".join(parser.text_content)
        print(f"Parsed {len(raw_text)} characters of raw text.")
        
        # If raw_text is empty, fallback to parsing all text (e.g. if bodyContent isn't found)
        if not raw_text.strip():
            print("Warning: bodyContent parser returned empty text. Falling back to body parsing...")
            parser = SimpleWikiParser()
            parser.in_body = True # Force recording
            parser.feed(html)
            raw_text = "".join(parser.text_content)
            print(f"Fallback parsed {len(raw_text)} characters of raw text.")
        
        # Clean Wikipedia specific artifacts:
        # 1. Remove Wikipedia citation brackets: [1], [12], [citation needed]
        text_no_citations = re.sub(r'\[\d+\]|\[citation needed\]|\[[a-zA-Z0-9\s,\-\.]+\]', '', raw_text)
        
        # 2. Remove editorial markers: [edit], [change]
        text_no_edits = re.sub(r'\[edit\]|\[change\]', '', text_no_citations)
        
        # 3. Clean up excessive whitespaces & blank lines
        lines = [line.strip() for line in text_no_edits.split('\n')]
        
        cleaned_lines = []
        for line in lines:
            # Exclude lines that are just navigation links or page headers
            if line.startswith("^") or line.startswith("Jump to:") or line.startswith("Main page"):
                continue
            if line:
                cleaned_lines.append(line)
                
        # Reconstruct text with double newlines between paragraphs
        final_text = "\n\n".join(cleaned_lines)
        
        # Perform minor formatting sanity checks
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)
        
        out_filename = "ligo_wikipedia.txt"
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write(final_text.strip())
            
        print(f"Successfully extracted LIGO Wikipedia content to: '{out_filename}'")
        print(f"File Size: {os.path.getsize(out_filename)} bytes")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
