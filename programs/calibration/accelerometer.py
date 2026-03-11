
import asyncio
import infra.utils as utils
import os
from datetime import datetime 

timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

num_seconds = 20
logfile = os.path.join("logs", "accelerometer", timestamp, "gravity_sphere_test.csv")

# Robot runs this program
def getRobotPrograms(robot):
    coro = utils.Calibrator(robot, num_seconds, logfile).calibrate()
    return [coro]



# Analyze the log - post calibration