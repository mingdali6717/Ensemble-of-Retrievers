import re
from abc import ABC, abstractmethod
import jsonlines
import os
import pickle
from typing import Iterable, List, Union

import torch
from collections import Counter
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
)

TScalar = Union[int, float, torch.Tensor]
TVector = Union[List[TScalar], torch.Tensor]
re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

DEFAULT_CORPUS_PATH = ["eor/evaluation/metrics/test.jsonl", "eor/evaluation/metrics/validation.jsonl", "eor/evaluation/metrics/train.jsonl"]

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s

class Metric(ABC):
    """
    Base class for storing metrics.

    Subclasses should define .value(). Examples are provided for each subclass.
    """

    @property
    def is_global(self) -> bool:
        """
        Indicates whether this metric should be reported globally or per-task.
        """
        return False

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return False

    @abstractmethod
    def value(self) -> float:
        """
        Return the value of the metric as a float.
        """
        pass

    @abstractmethod
    def __add__(self, other: Any) -> 'Metric':
        raise NotImplementedError

    def __iadd__(self, other):
        return self.__radd__(other)

    def __radd__(self, other: Any):
        if other is None:
            return self
        return self.__add__(other)

    def __str__(self) -> str:
        return f'{self.value():.4g}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value():.4g})'

    def __float__(self) -> float:
        return float(self.value())

    def __int__(self) -> int:
        return int(self.value())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() == other.value()
        else:
            return self.value() == other

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() < other.value()
        else:
            return self.value() < other

    def __sub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.
        """
        if not isinstance(other, float):
            raise TypeError('Metrics.__sub__ is intentionally limited to floats.')
        return self.value() - other

    def __rsub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.

        NOTE: This is not necessary in python 3.7+.
        """
        if not isinstance(other, float):
            raise TypeError('Metrics.__rsub__ is intentionally limited to floats.')
        return other - self.value()

    @classmethod
    def as_number(cls, obj: TScalar) -> Union[int, float]:
        if isinstance(obj, torch.Tensor):
            obj_as_number: Union[int, float] = obj.item()
        else:
            obj_as_number = obj  # type: ignore
        assert isinstance(obj_as_number, int) or isinstance(obj_as_number, float)
        return obj_as_number

    @classmethod
    def as_float(cls, obj: TScalar) -> float:
        return float(cls.as_number(obj))

    @classmethod
    def as_int(cls, obj: TScalar) -> int:
        return int(cls.as_number(obj))

    @classmethod
    def many(cls, *objs: List[TVector]) -> List['Metric']:
        """
        Construct many of a Metric from the base parts.

        Useful if you separately compute numerators and denominators, etc.
        """
        lengths = [len(o) for o in objs]
        objs = list(objs)  # convert from tuple for inplace modification
        for i, o in enumerate(objs):
            if isinstance(o, torch.Tensor):
                # if the tensor is on GPU, make sure we transfer the whole thing
                # at once, instead of one-element-at-a-time during our list
                # comprehension
                objs[i] = o.tolist()
        if len(set(lengths)) != 1:
            raise IndexError(f'Uneven {cls.__name__} constructions: {lengths}')
        return [cls(*items) for items in zip(*objs)]

    @classmethod
    def from_mask(
        cls, metric_per_token: torch.Tensor, mask: torch.Tensor
    ) -> List['Metric']:
        """
        From token-level metrics, returns an aggregate MyMetric per example in the
        batch.

        :param metric_per_token:
            a (batchsize x num_tokens) Tensor
        :param mask:
            a (batchsize x num_tokens) Tensor to mask out tokens that should *not* be considered in the aggregate metric calculation.
        :return:
            a (batchsize) Tensor
        """
        tokens_per_ex = mask.long().sum(dim=-1)
        metric_per_ex = (metric_per_token * mask).sum(dim=-1)
        metrics = cls.many(metric_per_ex, tokens_per_ex)
        return metrics

class AverageMetric(Metric):
    """
    Class that keeps a running average of some metric.

    Examples of AverageMetrics include hits@1, F1, accuracy, etc. These metrics all have
    per-example values that can be directly mapped back to a teacher.
    """

    __slots__ = ('_numer', '_denom')

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(self, numer: TScalar, denom: TScalar = 1):
        self._numer = self.as_number(numer)
        self._denom = self.as_number(denom)

    def __add__(self, other: Optional['AverageMetric']) -> 'AverageMetric':
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        full_numer: TScalar = self._numer + other._numer
        full_denom: TScalar = self._denom + other._denom
        # always keep the same return type
        return type(self)(numer=full_numer, denom=full_denom)

    def value(self) -> float:
        if self._numer == 0 and self._denom == 0:
            # don't nan out if we haven't counted anything
            return 0.0
        if self._denom == 0:
            return float('nan')
        return self._numer / self._denom

class F1Metric(AverageMetric):
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.

        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values

        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def compute(
        guess: str, answers: List[str], expose_p_and_r: bool = False
    ) -> Union['F1Metric', Tuple['F1Metric', 'F1Metric', 'F1Metric']]:
        if guess is None or answers is None:
            return AverageMetric(0, 0)
        g_tokens = normalize_answer(guess).split()
        scores = [
            F1Metric._prec_recall_f1_score(g_tokens, normalize_answer(a).split())
            for a in answers
        ]
        max_p, max_r, max_f1 = 0, 0, 0
        for p, r, f1 in scores:
            max_p, max_r, max_f1 = max(max_p, p), max(max_r, r), max(f1, max_f1)
        if expose_p_and_r:
            return (F1Metric(max_p, 1), F1Metric(max_r, 1), F1Metric(max_f1, 1))
        else:
            return F1Metric(max_f1, 1)

class RareWordF1Calculator:
    """
    Helper class for computing F1 with an emphasis on infrequent words.
    """

    def __init__(self, corpus: str = None, top_p: float = 0.5, corpus_path: Union[str, List[str]] = None ):
        try:
            import nltk
        except ImportError:
            raise ImportError('Please install nltk (e.g. pip install nltk).')
        
        if corpus_path is not None:
            corpus = construct_corpus(corpus_path)
        elif corpus is not None:
            assert isinstance(corpus, str) and not os.path.exists(corpus), "coupus should be the string of the whole corpus"
        else:
            corpus = construct_corpus(DEFAULT_CORPUS_PATH)

        words = normalize_answer(corpus).split()
        self._freq_dist = nltk.FreqDist(words)
        self._cutoff_count = RareWordF1Calculator._find_cutoff_count(
            self._freq_dist, top_p
        )

    @property
    def freq_dist(self):
        return self._freq_dist

    @staticmethod
    def _find_cutoff_count(freq_dist, top_p: float) -> int:
        """
        Finds the word occurance for which the cumulative occurances are `top_p` of the
        overall word count.
        """
        assert top_p < 1
        target = sum(freq_dist.values()) * top_p
        cumul = 0
        for _, v in freq_dist.most_common():
            cumul += v
            if cumul > target:
                return v
        raise RuntimeError(f"Invalid top {top_p*100}% of the corpus distribution")

    @staticmethod
    def _filter(freq_dist, cutoff: int, text: str) -> str:
        """
        For words that are found in the reference distribution, filters those with an
        occurrence count less than the cutoff.
        """
        words = normalize_answer(text).split()
        return " ".join([w for w in words if freq_dist.get(w, cutoff) < cutoff])

    def compute(self, guess: str, answers: Iterable[str]) -> F1Metric:
        if guess is None or answers is None:
            return F1Metric(0, 0)
        guess = RareWordF1Calculator._filter(self._freq_dist, self._cutoff_count, guess)
        answers = [
            RareWordF1Calculator._filter(self._freq_dist, self._cutoff_count, a)
            for a in answers
        ]
        if not any(len(a) for a in answers):
            # no rare words in labels, set denominator to zero
            return F1Metric(0, 0)
        return F1Metric.compute(guess, answers)
    
    def raref1_score(self, candidate, reference):
        return self.compute(candidate, [reference])

def construct_corpus(paths):
    if isinstance(str):
        paths = [paths]
    all_strings = []
    for path in paths:
        if os.path.isfile(path):

            with jsonlines.open(path) as reader:
                for item in reader:
                    all_strings.append(item["query"])
                    all_strings.extend(item["answer"])
        else:
            raise KeyError(f"given path '{path}' is invalid")
    corpus = " ".join(all_strings)
    return corpus









# guess = "This is a sample text containing onomatopoeiaasdasd."
# answers = ["This is a text with the word onomatopoeia.", "This text mentions xylophone."]
# score = raref1_score(guess, answers)
# print(score)


