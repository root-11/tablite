import logging
log = logging.getLogger()
log.setLevel(logging.INFO)



def test_loggers():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    tablite_logger_found = False
    for logger in loggers:
        if 'tablite' in logger.name:
            tablite_logger_found = True
        print(logger)
    assert tablite_logger_found    
