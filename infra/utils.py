
import importlib.util
import asyncio
import sys
from pathlib import Path
import os
import csv


# =====================================
# Command line args
# =====================================
def parseArgs():
        HELP=""
        import argparse
        import textwrap
        parser = argparse.ArgumentParser(
            formatter_class = argparse.RawTextHelpFormatter,
            description=textwrap.dedent(HELP)
        )
        parser.add_argument(
            "--program", type=str, help="python file containing list of co-routines", required=True
        )
        args = parser.parse_args()
        return args



# =================================================================
# Calibrator.
# Calibrate the robot for the specified duration,
# and dump its output in the specified logfile.
# Querry the robot's sensor at the specified sampling rate.
# =================================================================
class Calibrator:
    def __init__(self, 
                 robot, 
                 duration_seconds, 
                 logfile,
                 robot_sampling_frequency_hz = 10
                 ):
        self.robot                       = robot
        self.duration_seconds            = duration_seconds
        self.robot_sampling_frequency_hz = robot_sampling_frequency_hz
        self.stop_evt                    = asyncio.Event()

        # Create the outdir
        logfile2 = Path(logfile)
        logfile2.parent.mkdir(exist_ok=True, parents=True)
        self.outfile = open(logfile2, "w")
        self.writer  = csv.writer(self.outfile)
        self.writer.writerow(["time"] + list(range(20)) )

    # Calibrate.
    # Samples the robot for raw sensor data while the user
    # physically moves the robot certain parts of the robot 
    # to activate specific sensors.
    # This is more of an art than science.
    async def calibrate(self):  
        period     = 1.0 / self.robot_sampling_frequency_hz
        total_time = 0
        try:
            while not self.stop_evt.is_set() and total_time <= self.duration_seconds:
                s    = await self.robot.next_sensor()
                data = [s.t] + s.raw
                self.writer.writerow(data)
                self.outfile.flush()

                print(f"[logger] {s}")
                await asyncio.sleep(period)
                total_time = total_time + period
        
        except asyncio.CancelledError:
            raise



# ================================================================================
    # Dynamically imports a Python module specified by its name and file path.

    # Args:
    #     module_name (str): The name to assign to the imported module (can be arbitrary).
    #     file_path (str or Path): The full path to the .py file.

    # Returns:
    #     module: The imported module object, or None if an error occurred.
# ================================================================================
def loadModule(module_name, file_path):

    file_path = Path(file_path).resolve() # Ensure the path is absolute and resolved
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        print(f"Error: Could not create module specification for {file_path}")
        return None
    
    module = importlib.util.module_from_spec(spec)
    
    # Register the module with sys.modules to prevent re-import issues
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error executing module {module_name}: {e}")
        # Clean up the entry from sys.modules if execution fails
        del sys.modules[module_name]
        return None

