

class ExactMatches:
    def __init__(
        self,
        aggregation_function: callable = None,
        normalize_gold: callable = None,
        normalize_pred: callable = None,
        strip_strings: bool = False,
        type_exact_match: str = "full",
    ):
        """An exact match class.

        Args:
            aggregation_function (callable, optional): How to aggregate the item results. Defaults to max.
                Used if there are several golds or predictions on which scores were computed.
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
            strip_strings (bool, optional): Whether to strip both reference and predictions. Defaults to False.
            type_exact_match (str, optional): Defines what type of match to apply (post normalization if present).
                Can be any of `prefix`, `suffix` or `full`. Defaults to "full".
                `prefix` checks if the prediction starts with the gold,
                `suffix` if the prediction ends with the gold,
                `full` if the prediction and gold are equal
        """
        if aggregation_function is None:
            aggregation_function = max
        self.aggregation_function = aggregation_function
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
        self.strip_strings = strip_strings

        if type_exact_match not in ["prefix", "suffix", "full"]:
            # todo: we could add a set exact match
            raise ValueError(
                f"type_exact_match (used in parametrized_exact_match) must be one of prefix, suffix, or full. Was {type_exact_match} instead."
            )
        self.type_exact_match = type_exact_match

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float: Aggregated score over the current sample's items.
        """
        results = []
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The exact match score. Will be 1 for a match, 0 otherwise.
        """
        if not pred:
            return 0

        if self.strip_strings:
            gold = gold.strip()
            pred = pred.strip()

        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)

        if self.type_exact_match == "prefix":
            return 1 if pred.startswith(gold) else 0
        if self.type_exact_match == "suffix":
            return 1 if pred.endswith(gold) else 0
        return 1 if gold == pred else 0

