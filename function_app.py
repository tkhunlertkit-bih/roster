import logging
import os
import tempfile
from typing import Literal

import azure.functions as func

from file_storage.blob_storage_rw import BlobStorageRW
from file_storage.file_storage_interface import FileStorageInterface
from file_storage.share_point_rw import SharePointRW
from solver import solve

app = func.FunctionApp()


def get_storage_client(inout: Literal["input", "output"]) -> FileStorageInterface:
    storage_type = os.getenv("STORAGE_TYPE", "blob")

    if storage_type == "blob":
        return BlobStorageRW(
            connection_string=os.environ["AzureWebJobsStorage"],
            container_name=os.getenv(f"{inout.upper()}_CONTAINER", "roster"),
            folder_path=os.getenv(f"{inout.upper()}_FOLDER", "ipd_nurse"),
        )
    elif storage_type == "sharepoint":
        return SharePointRW(
            tenant_id=os.environ["GRAPH_TENANT_ID"],
            client_id=os.environ["GRAPH_CLIENT_ID"],
            client_secret=os.environ["GRAPH_CLIENT_SECRET"],
            site_id=os.environ["SHAREPOINT_SITE_ID"],
            folder_path=os.getenv(f"SHAREPOINT_{inout.upper()}_FOLDER_PATH", f"/SharedDocuments/RosterData/{inout}"),
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}. Avaliable types are 'blob' or 'sharepoint'")


@app.timer_trigger(schedule="0 0 5 * * *", arg_name="myTimer", run_on_startup=False)
def scheduled_function(myTimer: func.TimerRequest) -> None:
    """Timer-triggered roster solver"""
    logging.info("Scheduled roster solver started")
    try:
        input_storage = get_storage_client("input")
        output_storage = get_storage_client("output")

        # Read input files
        logging.info(f"Reading input files from storage: {os.environ['STORAGE_TYPE']}")
        nurses_df = input_storage.read_csv("nurses.csv")
        preferences_df = input_storage.read_csv("preferences.csv")
        beds_df = input_storage.read_csv("beds_per_day.csv")

        # Save to temp directory for solver
        with tempfile.TemporaryDirectory() as temp_dir:
            nurses_df.to_csv(os.path.join(temp_dir, "nurses.csv"), index=False)
            preferences_df.to_csv(os.path.join(temp_dir, "preferences.csv"), index=False)
            beds_df.to_csv(os.path.join(temp_dir, "beds_per_day.csv"), index=False)

        # Run solver
        logging.info("Running solver...")
        solve_parameters = {
            "config": os.environ["CONFIG_NAME"],
            "input_dir": temp_dir,
            "pub_days_per_nurse": int(os.environ["PUB_DAYS_PER_NURSE"]),
            "fte_uos_threshold": float(os.environ["FTE_UOS_THRESHOLD"]),
            "days": int(os.environ["DAYS"]),
            "max_time": float(os.getenv("MAX_TIME", 60)),
        }
        res = solve(**solve_parameters)  # Call your script logic
        logging.info(f"Solver completed for {os.environ['CONFIG_NAME']}")
        output_storage.write_excel("roster.xlsx", **res)
    except Exception as e:
        logging.error(f"Error in roster generation: {str(e)}", exc_info=True)
        raise


@app.route(route="run-solver", methods=["POST", "GET"], auth_level=func.AuthLevel.FUNCTION)
def manual_solver(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("Solver place holder", status_code=500)
