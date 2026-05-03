from __future__ import annotations

import re
from typing import Iterable, List


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÇĞİÖŞÜçğıöşü0-9]{2,}", re.UNICODE)

# Basic English + Turkish stopwords (compact set)
_STOPWORDS = {
    # English
    "the","a","an","and","or","but","if","then","else","when","at","by","for","in","of","on","to","with","is","are","was","were","be","been","being","as","it","its","this","that","these","those","from","into","about","over","after","before","between","out","up","down","so","than","too","very","can","cannot","may","might","should","would","could","will","just","not","no","nor","do","does","did","doing","have","has","had","having","you","your","yours","we","our","ours","they","them","their","theirs","he","him","his","she","her","hers","i","me","my","mine","all","any","both","each","few","more","most","other","some","such",
    # Turkish
    "ve","veya","ama","fakat","yalnız","ile","için","bir","bu","şu","o","da","de","ki","mi","mu","mü","mı","ile","gibi","daha","en","çok","az","her","hiç","çok","olan","olanlar","var","yok","ise","değil","ben","sen","o","biz","siz","onlar","şey","şeyi","hakkında","önce","sonra","içinde","üzerinde","altında","arası","arasında","kadar","göre","dolayı","nedenle","ama","çünkü","ile","değil","ne","nasıl","nerede","neden","hangi",
}


def tokenize(text: str, *, remove_stopwords: bool = True) -> List[str]:
    if not text:
        return []
    text = text.casefold()
    tokens = _TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


