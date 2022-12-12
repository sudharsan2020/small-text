import logging

from collections import OrderedDict
from pathlib import Path

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.annotations import deprecated


logger = logging.getLogger(__name__)


try:
    import torch
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


@deprecated(deprecated_in='1.1.0', to_be_removed_in='2.0.0')
class Metric(object):

    def __init__(self, name, lower_is_better=True):
        """
        Represents any metric.

        Parameters
        ----------
        name : str
            A name for the metric.
        lower_is_better : bool
            Indicates if the metric is better for lower values if True,
            otherwise it is assumed that higher values are better.
        """
        self.name = name
        self.lower_is_better = lower_is_better


@deprecated(deprecated_in='1.1.0', to_be_removed_in='2.0.0')
class PytorchModelSelection(object):

    IDX_EPOCH = -2

    def __init__(self, save_directory, metrics, sort_by_idx=0):
        if isinstance(save_directory, Path):
            self.save_directory = save_directory
        else:
            self.save_directory = Path(save_directory)

        self.sort_by_idx = sort_by_idx

        # (result_metric1, result_metric2, ..., result_metric_n, epoch, model_nr) -> model path
        self.metrics = OrderedDict({metric.name: metric for metric in metrics})
        self.models = OrderedDict()

    def add_model(self, model, epoch, **kwargs):

        if any(metric_name not in kwargs for metric_name in self.metrics.keys()):
            raise ValueError('All metrics defined in the constructor must be reported. '
                             'Expected metrics: ' + ', '.join(self.metrics.keys()))

        model_path = Path(self.save_directory).joinpath(f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_path)
        model_id = tuple(kwargs[metric.name] for metric in self.metrics.values()) \
                + (epoch, len(self.models))
        self.models[model_id] = model_path

    def select_best(self):
        models = sorted(self.models.items(), key=lambda x: self._get_sort_key(x))
        first_metric = list(self.metrics.items())[0][1]
        model_number = models[0][0][-1]
        logger.info('Using model {} ({}={:1f})'.format(model_number+1,
                                                       first_metric.name,
                                                       models[0][0][0]))
        self.selected_model = model_number
        return models[0][1], models[0][0]

    def select_last(self):
        model_number = len(self.models)
        keys = list(self.models.keys())
        last_model_key = keys[model_number-1]

        logger.info(f'Using last model {model_number}')
        self.selected_model = model_number

        return self.models[last_model_key], last_model_key

    def _get_sort_key(self, x):
        data = []
        for idx, metric in enumerate(self.metrics.values()):
            data += [x[0][idx]] if metric.lower_is_better else [-x[0][idx]]
        data += [x[0][self.IDX_EPOCH]]

        return tuple(data)


@deprecated(deprecated_in='1.1.0', to_be_removed_in='2.0.0')
def validate_metrics(metrics):

    if not isinstance(metrics, list):
        raise ValueError('Argument "metrics" must be a list')

    for metric in metrics:
        if not isinstance(metric, Metric):
            raise ValueError(f'Invalid metric: "{str(metric)}" ({type(metric)})')
