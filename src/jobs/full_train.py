import torch
import logging

logger = logging.getLogger(__name__)


def full_train():
    logger.info("CUDA availability: %s", torch.cuda.is_available())
    logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))
