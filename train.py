from pl_model import YOLOv4PL
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

m = YOLOv4PL()
m.cpu();
tb_logger = pl.loggers.TensorBoardLogger('logs/', name="yolov4")
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_last=True,
    monitor='training_loss_epoch',
    dirpath='model_checkpoints/',
    filename='yolov4-{epoch:02d}-{val_loss:.2f}',
    mode='min',
)

t = pl.Trainer(default_root_dir='checkpoints',
               logger=tb_logger,
               gpus=1,
               precision=32,
               benchmark=True,
               callbacks=[checkpoint_callback, LearningRateMonitor()],
               min_epochs=100,

               #            resume_from_checkpoint="model_checkpoints/yolov4epoch=82.ckpt",
               #    auto_lr_find=True,
               #  auto_scale_batch_size='binsearch',
               #    fast_dev_run=True
               )
r = t.tuner.lr_find(m, min_lr=1e-10, max_lr=1, early_stop_threshold=None)
r.plot()
t.fit(m)
