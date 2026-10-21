from typing import List


class Configuration(object):
    learning_rate_list: List[float]
    number_of_centers_list: List[int]
    seeds: List[int]
    def __init__(
            self,
            learning_rate_list: List[float],
            number_of_centers_list: List[int],
            seeds_list: List[int]
    ) -> None:
        self.learning_rate_list = learning_rate_list
        self.number_of_centers_list = number_of_centers_list
        self.seeds = seeds_list

    def __str__(self) -> str:
        total = (
                len(self.learning_rate_list)
                * len(self.number_of_centers_list)
                * len(self.seeds)
        )
        return (
            "Configuration(\n"
            f"  learning_rates: {self.learning_rate_list}\n"
            f"  centers:        {self.number_of_centers_list}\n"
            f"  seeds:          {self.seeds}\n"
            f"  total runs:     {total}\n"
            ")"
        )

    __repr__ = __str__
