# class BPI17OCEL(TUEventLog):
#     """BPI Challenge 2017 OCEL

#     DOI: 10.4121/6889ca3f-97cf-459a-b630-3b0b0d8664b5.v1

#     This dataset contains the result of transforming the BPI Challenge 2017
#     event log from Event Graph format to Object-Centric Event Log (OCEL) format.
#     The transformation is defined in the "Transforming Event Knowledge Graph
#     to Object-Centric Event Logs: A Comparative Study for Multi-dimensional
#     Process Analysis" paper.

#     Args:
#         root_path (str, optional): Path where the event log will be stored.
#             Defaults to "./data".
#         train (bool, optional): If True, returns the training set. Otherwise,
#             returns the test set. Defaults to True.
#         kwargs: Additional arguments to be passed to the base class.
#     """

#     url = "https://data.4tu.nl/file/6889ca3f-97cf-459a-b630-3b0b0d8664b5/5d5b9f89-7fa6-4c92-b6ac-04f854bdf92e"

#     def __init__(
#         self, root_path: str = "./data", train: bool = True, **kwargs: Any
#     ) -> None:
#         super().__init__(root_path=root_path, **kwargs)
#         self.train = train

#         if not self._check_raw():
#             self.download(self.url)

#     def _load_log(self):
#         raise NotImplementedError

#     @property
#     def raw_log(self):
#         return os.path.join(self.raw_folder, f"{self.__class__.__name__}.jsonocel")
