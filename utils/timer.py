import time 
from contextlib import contextmanager


@contextmanager
def simple_timer(name=''):
    t = time.time()
    yield 
    print(f"{name} exec time: {time.time() - t}")


def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class timming(object):
    def __init__(self):
        super().__init__()
        self.items={}

    def start(self, item_name):
        if item_name not in self.items:
            self.items[item_name]={
                "end_cnt": 0,
                "avg": 0,
                "start_t": time.time(),
                "is_finished": False
            }
        else:
            assert self.items[item_name]["is_finished"]==True
            self.items[item_name].update(
                {   
                    # "start_cnt":  self.items[item_name]["start_cnt"]+1,
                    "end_cnt": self.items[item_name]["end_cnt"], #unchanged
                    "avg": self.items[item_name]["avg"], #unchanged 
                    "start_t": time.time(),
                    "is_finished": False
                }
            )

    def end(self, item_name):
        assert item_name in self.items
        t=time.time()
        interval=t-self.items[item_name]['start_t']
        self.items[item_name].update(
            {   
                "end_cnt":  self.items[item_name]["end_cnt"]+1,
                "avg": (self.items[item_name]["avg"]*self.items[item_name]["end_cnt"]+interval)/(self.items[item_name]["end_cnt"]+1) ,
                "is_finished": True
            }
        )
    def summarize(self):
        for k in self.items:
            print(f"Average time of {k} = {self.items[k]['avg']*1000} ms", f"(averaged on {self.items[k]['end_cnt']} testing cycles.") 
