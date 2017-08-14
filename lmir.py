# Michael A. Alcorn (malcorn@redhat.com)
# See: http://sifaka.cs.uiuc.edu/czhai/pub/sigir2001-smooth.pdf.


class LMIR:
    def __init__(self, corpus, lamb=0.1, mu=2000, delta=0.7):
        """Use language models to score query/document pairs.

        :param corpus: 
        :param lamb: 
        :param mu: 
        :param delta: 
        """
        self.lamb = lamb
        self.mu = mu
        self.delta = delta

        # Fetch all of the necessary quantities for the document language
        # models.
        doc_token_counts = []
        doc_lens = []
        doc_p_mls = []
        all_token_counts = {}
        for doc in corpus:
            doc_len = len(doc)
            doc_lens.append(doc_len)
            token_counts = {}
            for token in doc:
                token_counts[token] = token_counts.get(token, 0) + 1
                all_token_counts[token] = token_counts.get(token, 0) + 1

            doc_token_counts.append(token_counts)

            p_ml = {}
            for token in token_counts:
                p_ml[token] = token_counts[token] / doc_len

        total_tokens = sum(all_token_counts.values())
        p_C = {token: token_count / total_tokens
               for (token, token_count) in all_token_counts.items()}

        self.N = len(corpus)
        self.c = doc_token_counts
        self.doc_lens = doc_lens
        self.p_ml = doc_p_mls
        self.p_C = p_C

    def jelinek_mercer(self, query_tokens):
        """Calculate the Jelinek-Mercer scores for a given query.

        :param query_tokens: 
        :return: 
        """
        lamb = self.lamb
        p_C = self.p_C
        scores = []
        for doc_idx in range(self.N):
            p_ml = self.p_ml[doc_idx]
            score = 0
            for token in query_tokens:
                score += (1 - lamb) * p_ml[token] + lamb * p_C[token]

            scores.append(score)

        return scores

    def dirichlet(self, query_tokens):
        """Calculate the Dirichlet scores for a given query.

        :param query_tokens: 
        :return: 
        """
        mu = self.mu
        p_C = self.p_C

        scores = []
        for doc_idx in range(self.N):
            c = self.c[doc_idx]
            doc_len = self.doc_lens[doc_idx]
            score = 0
            for token in query_tokens:
                score += (c[token] + mu * p_C[token]) / (doc_len + mu)

            scores.append(score)

        return scores

    def absolute_discount(self, query_tokens):
        """Calculate the absolute discount scores for a given query.

        :param query_tokens: 
        :return: 
        """
        delta = self.delta
        p_C = self.p_C

        scores = []
        for doc_idx in range(self.N):
            c = self.c[doc_idx]
            doc_len = self.doc_lens[doc_idx]
            d_u = len(c)
            score = 0
            for token in query_tokens:
                score += max(c[token] - delta, 0) / doc_len + delta * d_u / doc_len * p_C[token]

            scores.append(score)

        return scores
