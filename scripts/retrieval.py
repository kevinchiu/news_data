"""Shared sparse BM25 retrieval over the sentence corpus."""
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse

STOP = set("the a an of to in and for on at by with from as is was were be been it its that this said would will has have had pct mln dlrs".split())


def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s.lower()) if t not in STOP]



class BM25:
    """Okapi BM25 over a sparse term matrix (rank_bm25 is too slow for 150k sentences)."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.vocab = {}
        rows, cols, vals = [], [], []
        for i, doc in enumerate(corpus):
            counts = {}
            for t in doc:
                j = self.vocab.setdefault(t, len(self.vocab))
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i); cols.append(j); vals.append(c)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(corpus), len(self.vocab)), dtype=np.float32)
        dl = np.array([len(d) for d in corpus], dtype=np.float32)
        df = np.bincount(cols, minlength=len(self.vocab))
        self.idf = np.log((len(corpus) - df + 0.5) / (df + 0.5) + 1).astype(np.float32)
        norm = k1 * (1 - b + b * dl / dl.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.w = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def get_scores(self, query):
        ids = [self.vocab[t] for t in set(query) if t in self.vocab]
        return np.asarray(self.w[:, ids] @ self.idf[ids]).ravel()


class SentenceIndex:
    """Sentences + story headlines with BM25 search restricted to strictly earlier stories."""

    def __init__(self, sentences_path, articles_path):
        self.rows = [json.loads(l) for l in open(sentences_path)]
        self.headlines = {}
        for l in open(articles_path):
            a = json.loads(l)
            self.headlines[a["id"]] = a["title"] or (a["body"] or "")[:120]
        self.bm25 = BM25([toks(r["text"]) for r in self.rows])
        self.dates = np.array([r["date"] for r in self.rows])
        self.docs = np.array([r["doc_id"] for r in self.rows])
        self.by_sid = {r["sid"]: i for i, r in enumerate(self.rows)}

    def headline(self, doc_id):
        return self.headlines.get(doc_id, "")

    def search(self, text, before, k=30, exclude_docs=()):
        s = self.bm25.get_scores(toks(text))
        mask = self.dates >= before
        for d in exclude_docs:
            mask |= self.docs == d
        s[mask] = -1
        top = np.argsort(-s)[:k]
        return [dict(self.rows[j], score=float(s[j]), headline=self.headline(self.rows[j]["doc_id"]))
                for j in top if s[j] > 0]
