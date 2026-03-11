import infra.utils as utils
import infra.dash_driver as dash_driver
import asyncio

# Load command line
args = utils.parseArgs()

# Load the program file
program_file = args.program
program = utils.loadModule("program", program_file)

# Create the robot interface
robot  = dash_driver.DashRobot("Dash")

# Ensure that the co-routines exist
coro_list = program.getRobotPrograms(robot)


# ====================================================================
# Orchestrate the session
# ====================================================================
async def run_session() -> None:

    """
    One full session:
    1) connect
    2) auto-start PHY + safety inside robot.connect()
    3) run motion + logger in parallel
    4) graceful shutdown
    """
    session_stop_evt = asyncio.Event()

    try:
        # T1: serial setup
        await robot.connect(timeout=6.0)
        print("Robot found and connected")

        # T4/T5: foreground tasks in parallel
        await asyncio.gather( *coro_list)

    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
        session_stop_evt.set()
        raise

    finally:
        # T6: graceful shutdown
        session_stop_evt.set()
        try:
            await robot.stop()
        except Exception:
            pass
        await robot.close()
        print("Robot closed cleanly")


# Launch
asyncio.run(run_session())
