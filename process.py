import logging, sys, sklearn

def process_frame(frame):
    if len(frame.hands) != 1:
        logging.error("Incorrect number of hands, bad frame")
        return
