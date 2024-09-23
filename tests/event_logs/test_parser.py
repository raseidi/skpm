# from skpm.event_logs.parser import read_xes
# from skpm.event_logs import (
#     BPI12,
#     BPI13ClosedProblems,
#     BPI13Incidents,
#     BPI17,
#     BPI19,
#     BPI20,
# ) 

def test_rerad_xes():
    assert True
#     """ToDo
    
#     I gotta learn how to cache files on GitHub Actions.
#     """
    
#     logs = (
#         BPI12,
#         BPI13ClosedProblems,
#         BPI13Incidents,
#         BPI17,
#         BPI19,
#         # BPI20,
#     )

#     shapes = {
#         "BPI12": (262200, 7),
#         "BPI13ClosedProblems": (6660, 12),
#         "BPI13Incidents": (65533, 12),
#         "BPI17": (1202267, 19),
#         "BPI19": (1595923, 21),
#     }
    
#     for l in logs:
#         df = l()        
#         assert df.log.shape == shapes[l.__name__]