import os
from tensorboardX import SummaryWriter


class MetricManager:
    """Manages updating and logging of metrics"""
    def __init__(self, log_dir, metrics=None, print_metrics=True,
                 train_prefix='train/', val_prefix='val/', **writer_kwargs):
        self._validate_metrics(metrics)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = metrics
        self.writer = SummaryWriter(logdir=log_dir, **writer_kwargs)
        self.training = True
        self.print_metrics = print_metrics
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix

    def train(self, training=True):
        """Set the training mode of the manager"""
        if training != self.training:
            self.reset()  # reset metrics if we are switching modes
        self.training = training

    def eval(self):
        """Set the manager to evaluation mode"""
        self.train(False)

    def update(self, data_dict):
        """Update running values"""
        for metric in self.metrics:
            metric.update(data_dict)

    def reset(self):
        """Reset running values"""
        for metric in self.metrics:
            metric.reset()

    def log(self, step, epoch=None):
        """Log the current values"""
        # Initialize log string if required
        if self.print_metrics:
            label = 'Training' if self.training else 'Eval'
            if epoch is not None:
                log_str = '[({}) epoch: {}, step: {}'.format(label, epoch, step)
            else:
                log_str = '[({}) step: {}'.format(label, step)

        # Get tag prefix for current mode
        tag_prefix = self.train_prefix if self.training else self.val_prefix
        print_ = False
        for metric in self.metrics:
            # Metrics are logged if we are in eval mode or if we are in training mode and are at a logging step
            if not self.training or (self.training and not step % metric.log_interval):
                metric.log(self.writer, step, tag_prefix)
                if self.print_metrics:
                    value = metric.value
                    if value is not None:
                        log_str += ', {}: {:.3f}'.format(metric.name, value)
                metric.reset()
                print_ = True
        if self.print_metrics and print_:
            print(log_str + ']', flush=True)

    def flush(self):
        """Flush the SummaryWriter queue to ensure all data is logged"""
        self.writer.flush()

    @property
    def values(self):
        """Computes final metric values"""
        values = {}
        for metric in self.metrics:
            value = metric.value
            if value is not None:
                values[metric.name] = metric.value
        return values

    @staticmethod
    def _validate_metrics(metrics):
        """Ensures that all metrics are instances of Metric"""
        if metrics is not None:
            for metric in metrics:
                if not isinstance(metric, Metric):
                    raise ValueError('All metrics must be instances of Metric')


class Metric:
    """Base class for all metrics"""
    def __init__(self, name, log_interval):
        self.name = name
        self.log_interval = log_interval

    def update(self, data_dict):
        """Update running values"""
        raise NotImplementedError

    def reset(self):
        """Reset running values"""
        raise NotImplementedError

    def log(self, writer, step, tag_prefix='val/'):
        """Log the current value(s)"""
        raise NotImplementedError

    @property
    def value(self):
        """Compute final metric value"""
        raise NotImplementedError


class ScalarMetric(Metric):
    """Metric which tracks the average value of a scalar"""
    def __init__(self, name, log_interval, scalar_key):
        super().__init__(name, log_interval)
        self.scalar_key = scalar_key
        self.scalar_sum = 0
        self.scalar_count = 0

    def update(self, data_dict):
        self.scalar_sum += data_dict[self.scalar_key].item()
        self.scalar_count += 1

    def reset(self):
        self.scalar_sum = 0
        self.scalar_count = 0

    def log(self, writer, step, tag_prefix=''):
        writer.add_scalar(tag_prefix + self.name, self.value, step)

    @property
    def value(self):
        return self.scalar_sum/self.scalar_count
