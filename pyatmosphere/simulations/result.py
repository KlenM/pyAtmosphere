import numpy as np
import pandas as pd


class Result:
    save_float_format = '{:.3e}'.format

    def __init__(self, channel, measures, max_size=None, save_path: str = ""):
        self.channel = channel
        self.measures = measures
        self.set_max_size(max_size)
        self.save_path = save_path
        if self.save_path:
            try:
                self.load_output()
                print(f"Loaded measures from {self.save_path}")
            except FileNotFoundError:
                pass

    def set_max_size(self, max_size):
        for measures in self.measures:
            measures.max_size = max_size

    def print_output(self):
        print(f"Len of the first measures: {len(self.measures[0])}")

    def plot_output(self):
        self.print_output()

    def as_df(self):
        df = pd.DataFrame([measures.data for measures in self.measures]).T
        df.columns = [self.measures[i].name for i in range(len(self.measures))]
        return df

    def save_output(self):
        if not self.save_path:
            return
        self.as_df().to_csv(self.save_path, index=False,
                            float_format=self.save_float_format)

    def load_output(self):
        for measures, data in zip(self.measures, pd.read_csv(self.save_path).T.values):
            measures.data = data.tolist()


#   def save_output(self):
#     if path_str := self.save_kwargs.get("path", None):
#       if self.save_kwargs.get("pickle", False):
#         self.channel.output = None
#         for ps in self.channel.path.phase_screens:
#           ps.cache_clear()
#         path = Path(path_str)
#         path.parent.mkdir(parents=True, exist_ok=True)
#         with path.open("wb") as f:
#           pickle.dump(self, f)
#       else:
#         raise NotImplementedError

#   @classmethod
#   def load_output(cls, path_str, is_pickle=False):
#     if is_pickle:
#       path = Path(path_str)
#       with path.open("rb") as f:
#         return pickle.load(f)
#     else:
#       raise NotImplementedError

#   def save_output(self):
#     with open(self.save_path, "w") as f:
#       f.write(",".join([measures.name for measures in self.measures]) + "\n")
#       for iteration_measures in zip(*[measures.data for measures in self.measures]):
#         f.write(",".join((f"{value:.3e}" for value in iteration_measures)) + "\n")

#   def load_output(self):
#     with open(self.save_path, "r") as f:
#       header = f.readline().split(",")
#       for measures in self.measures:
#         measures.data = []
#       for line in f.readlines():
#         for measures, data in zip(self.measures, line.split(",")):
#           measures.data.append(float(data))
