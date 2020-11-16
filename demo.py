#!/usr/bin/env python3

from absl import app, flags, logging
from absl.logging import PythonFormatter
from logging import LogRecord
from typing import Dict

import sh

import torch as th
import pytorch_lightning as pl

import nlp
import transformers

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 3e-3, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_integer('percent', 5, '')

FLAGS = flags.FLAGS

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')


class CustomPythonFormatter(PythonFormatter):
    """Logging Formatter to add colors and custom format string

    adapted from https://stackoverflow.com/q/64263526 (author: MaKaNu)
    """

    blue = "\x1b[34;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    grey = "\x1b[38;5;246m"
    reset_color = "\x1b[0m"
    format_str = "%(levelname)s" + reset_color + " - %(funcName)s()\n »»» " + grey + \
                 " %(message)s \n" + reset_color
    date_format = "%Y-%m-%d %H:%M:%S"

    FORMATS = {
        "DEBUG": logging.PythonFormatter(fmt=blue + format_str, datefmt=date_format),
        "INFO": logging.PythonFormatter(fmt=green + format_str, datefmt=date_format),
        "WARNING": logging.PythonFormatter(fmt=yellow + format_str, datefmt=date_format),
        "ERROR": logging.PythonFormatter(fmt=red + format_str, datefmt=date_format),
        "FATAL": logging.PythonFormatter(fmt=bold_red + format_str, datefmt=date_format)
    }

    def format(self, record: LogRecord):
        formatter = self.FORMATS.get(record.levelname, self.FORMATS["INFO"])
        return formatter.format(record)


class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')

        # do not optimize weights in the BERT encoder, only in the classification 'head' network
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)

        def _tokenize(x: nlp.Dataset):
            x['input_ids'] = tokenizer.batch_encode_plus(
                x['text'],
                max_length=FLAGS.seq_length,
                pad_to_max_length=True
            )['input_ids']
            return x

        def _prepare_ds(split: str):
            size_to_use = FLAGS.batch_size if FLAGS.debug else "100%"
            ds = nlp.load_dataset('imdb', split=f'{split}[:{size_to_use}]') \
                .shuffle(seed=42)
            logging.debug(f"ds[0] = {ds[0]}")
            logging.info(f"ds.dataset_size = {ds.dataset_size}")

            if FLAGS.percent < 100:
                logging.info(f"filtering ds to get only {FLAGS.percent}% of data")
                ds = ds.train_test_split(test_size=FLAGS.percent / 100, shuffle=False)['test']
                logging.info(f"after filtering ds.dataset_size = {ds.dataset_size}")

            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.test_ds = map(_prepare_ds, ('train', 'test'))

    def forward(self, input_ids: th.Tensor):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits

    def training_step(self, batch: Dict[str, th.Tensor], batch_idx: int):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch: Dict[str, th.Tensor], batch_idx: int):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs: Dict[str, th.Tensor]):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}

    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.train_ds,
            batch_size=FLAGS.batch_size,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.test_ds,
            batch_size=FLAGS.batch_size,
            drop_last=False,
            shuffle=True,
        )

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
        )


def main(_):
    logging.get_absl_handler().setFormatter(CustomPythonFormatter())
    logging.set_verbosity(logging.DEBUG)

    model = IMDBSentimentClassifier()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='imdb', version=0),
    )
    trainer.fit(model)


if __name__ == '__main__':
    app.run(main)
