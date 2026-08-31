import logging
import os


def get_logger(
    name="train",
    log_dir="logs",
):
    os.makedirs(
        log_dir,
        exist_ok=True,
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 防止重复添加 Handler
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    # 写入日志文件
    file_handler = logging.FileHandler(
        os.path.join(
            log_dir,
            "train.log",
        )
    )
    file_handler.setFormatter(formatter)
    # 输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Logger test successful.")