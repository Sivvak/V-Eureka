from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import time


def load_tensorboard_logs(path):
    # TODO this is hotfix, so that tensorboard_logs get saved in time,
    # maybe a better solution exists
    time.sleep(2)

    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)

    data = {
        k.split('/', 1)[1] if k.startswith("Episode/") else k: v
        for k, v in data.items()
    }

    return data
