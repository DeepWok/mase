def rlog(logger, rank, msg, level="info"):
    """
    Only log on rank 0 to avoid repeated messages.
    """
    log_fn = getattr(logger, level, logger.info)
    if rank == 0:
        log_fn(msg)
