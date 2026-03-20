import re

def clean_text(text):
    # Agar text string nahi hai (jaise NaN/null), toh usko string bana do
    if not isinstance(text, str):
        text = str(text)
        
    # Sab kuch lowercase (chote aksharo) mein kardo
    text = text.lower()
    
    # Faltu symbols, links, aur emojis ko hata do (sirf alphabets aur numbers rakho)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Extra spaces ko hata kar ek single space bana do
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
