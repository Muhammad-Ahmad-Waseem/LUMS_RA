import tracker
from detector import Detector
import cv2
import time
import datetime
import sys
import os

from utils.db.dbconfig import config_db, connect_db, creat_table, insert_data_db, close_connection

if __name__ == '__main__':
    img_size1 = 640
    img_size2 = 640

    # Initialize yolov5
    detector = Detector()
    names = detector.names
    trk_ids = {k: [] for k in names}
    trk_counts = {k: 0 for k in names}

    source = os.getenv('STREAM_SOURCE')  # from where to load video
    time_step = os.getenv('TIME_SLAB')  # the time step in minutes

    assert source is not None, "Video path not set"
    assert time_step is not None, "Time slab is not defined"
    time_step = float(time_step)
    frame_count = 0

    capture = cv2.VideoCapture(source)
    time_start = time.time()

    # database parameters
    db_params = config_db("utils/db/dbconfig.INI", "DEV")
    db_connection = connect_db(db_params)
    assert db_connection is not None, "Cannot connect to Database"
    db_cursor = creat_table(db_connection)

    while True:
        # Read each frame of the picture
        resolution_str = None
        _, im = capture.read()
        if im is None:
            break
        frame_count = frame_count + 1
        resolution = im.shape
        resolution_str = str(resolution[1]) + 'x' + str(resolution[0])
        im = cv2.resize(im, (img_size1, img_size2))
        list_bboxs = []
        bboxes = detector.detect(im)

        # If there is a bbox in the screen
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)
            tracks = [track for track in list_bboxs if track[4] in names]
            for track in tracks:
                class_name = track[4]
                other_clss = names.copy()
                other_clss.remove(class_name)
                track_id = (track[5])
                if track_id not in trk_ids[class_name]:
                    trk_ids[class_name].append(track_id)
                    trk_counts[class_name] = trk_counts[class_name] + 1
                else:
                    for entry in other_clss:
                        if track_id in trk_ids[entry]:
                            trk_ids[entry].remove(track_id)
        else:
            pass

        pass
          
        end = time.time()
        seconds = end - time_start

        if seconds > time_step * 60:
            #print("track counts before zeroing: ", trk_counts)
            fps = frame_count / seconds
            print("Estimated frames per second : {0}".format(fps))
            # (Cam_ID, cam_resolution, time_step, record_time, car, motorcycle, van, rickshaw, bus, truck)
            insert_data_db(db_connection, db_cursor, source, resolution_str, time_step, datetime.datetime.now(),
                           trk_counts['car'], trk_counts['motorcycle'], trk_counts['van'],
                           trk_counts['rickshaw'], trk_counts['bus'], trk_counts['truck'])
            time_start = time.time()
            frame_count = 0
            #trk_ids = {k: [] for k in names}
            trk_counts = {k: 0 for k in names}
            #print("track counts after zeroing: ", trk_counts)

    capture.release()
    '''
    if frame_count > 0:
        fps = frame_count / time.time()
        print("Estimated frames per second : {0}".format(fps))
        insert_data_db(db_connection, db_cursor, source, resolution_str, time_step, datetime.datetime.now(),
                       trk_counts["car"], trk_counts['motorcycle'], trk_counts['van'],
                       trk_counts['rickshaw'], trk_counts['bus'], trk_counts['truck'])'''

    close_connection(db_connection)

