import logging
import os

import azure.functions as func

from solver import solve

app = func.FunctionApp()


@app.timer_trigger(schedule="0 */10 * * * *", arg_name="myTimer", run_on_startup=False)
def scheduled_function(myTimer: func.TimerRequest) -> None:
    """Timer-triggered roster solver"""
    logging.info("Scheduled roster solver started")
    solve_parameters = {
        "config": os.getenv("CONFIG_NAME", ""),
        "input_dir": os.getenv("INPUT_DIR", ""),
        "pub_days_per_nurse": int(os.getenv("PUB_DAYS_PER_NURSE", 0)),
        "fte_uos_threshold": float(os.getenv("FTE_UOS_THRESHOLD", 0)),
        "days": int(os.getenv("DAYS", 0)),
        "max_time": float(os.getenv("MAX_TIME", 60)),
    }
    solve(**solve_parameters)  # Call your script logic


@app.route(route="run-solver", methods=["POST", "GET"], auth_level=func.AuthLevel.FUNCTION)
def manual_solver(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("Solver place holder", status_code=500)
