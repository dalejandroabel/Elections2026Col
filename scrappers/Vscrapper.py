import os
import requests
import pandas as pd
import orjson
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from E14 import E14Extractor
from dotenv import load_dotenv
import time
load_dotenv()


ERROR_LOG_FILE = os.getenv("ERROR_LOG_FILE")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
VALEROS_FILE = os.getenv("VALEROS_FILE")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WORKERS = min(32, os.cpu_count() * 2)
REQUEST_TIMEOUT = (27,230)
RETRIES_ALLOWED = 5
BACKOFF_FACTOR = 0.5

FILE_INIT = "data/esc/v1/actas-documentos/001/"
BASE_URL = "https://escrutiniospresidente2026.registraduria.gov.co"

local_thread = threading.local()

def get_session():
    """ Creates a thread-local session with retry logic for HTTP requests.

    Returns:
        requests.Session: A session object configured with retry logic.
    """

    if not hasattr(local_thread, "session"):

        session = requests.Session()

        retry = Retry(
            total=RETRIES_ALLOWED,
            connect=RETRIES_ALLOWED,
            read=RETRIES_ALLOWED,
            status=RETRIES_ALLOWED,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )

        adapter = HTTPAdapter(
            pool_connections=MAX_WORKERS,
            pool_maxsize=MAX_WORKERS,
            max_retries=retry,
        )

        session.mount("https://", adapter)

        local_thread.session = session

    return local_thread.session

def parse_location(dir_path: str) -> list[str]:
    """Extract [Department, Municipality, Zone, Location] from a path key."""
    return (
        dir_path
        .replace(FILE_INIT, "")
        .replace("/mesas/", "")
        .split("/")
    )

def make_output_path(department: str, municipality: str) -> str:
    """Return the full path for a Department-Municipality output file."""
    return os.path.join(OUTPUT_DIR, f"D{department}_M{municipality}.jsonl")

def append_zone_to_file(path: str, zone_key: str, zone_data: dict) -> None:
    """Append a single zone record as one JSON line to the output file.
    
    Each line has the shape: {"zone": <zone>, "locations": {<location>: {<table_no>: {...}}}}
    Using JSONL (one record per line) means we never need to hold the whole
    municipality in memory and partial writes survive crashes.
    """
    record = {"zone": zone_key, "locations": zone_data}
    with open(path, "ab") as f:          # binary mode — orjson returns bytes
        f.write(orjson.dumps(record) + b"\n")

def process_table(table_path: str) -> dict:
    """Download a PDF and extract votes.  Returns a result dict.

    `table_path` is the relative path segment returned by the listing API,
    e.g. '/data/esc/v1/actas-documentos/001/.../acta.pdf'
    The full URL is BASE_URL + table_path.
    """
    pdf_url = BASE_URL + table_path
    last_exc = None
    for attempt in range(RETRIES_ALLOWED):
        try:
            session = get_session()
            response = session.get(pdf_url, stream=True, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            pdf_bytes = response.content     
            declared = response.headers.get("Content-Length")
            if declared and len(pdf_bytes) < int(declared):
                raise IOError(
                    f"Truncated download: got {len(pdf_bytes)} of {declared} bytes"
                )
            extractor = E14Extractor(pdf_bytes,canvass_type="V", verbose=False, render_scale = 3)
            return {
                "success": True,
                "filename": table_path,
                "votes": extractor.resolve_as_json(),
            }
        except Exception as exc:
            last_exc = exc
            if attempt < RETRIES_ALLOWED - 1:
                time.sleep(BACKOFF_FACTOR * (2 ** (attempt - 1)))  # exponential backoff
    return {
                "success": False,
                "filename": table_path,
                "error": str(last_exc),
        }

def scrape() -> None:

    with open(VALEROS_FILE, "rb") as fh:
        json_data: dict = orjson.loads(fh.read())

    records = [
        {"Department": parts[0], "Municipality": parts[1],
         "Zone": parts[2],       "Location": parts[3],
         "listing_file": listing}
        for raw_key, listing in json_data.items()
        for parts in [parse_location(raw_key)]
    ]
    df = pd.DataFrame(records)

    with (
        open(ERROR_LOG_FILE, "a") as error_file):
        departments = df["Department"].unique()
        for department in tqdm(departments, desc="Departments", unit="dept"):
            dept_df = df[df["Department"] == department]

            municipalities = dept_df["Municipality"].unique()
            for municipality in tqdm(municipalities, desc="  Municipalities",
                                     unit="muni", leave=False):
                muni_df = dept_df[dept_df["Municipality"] == municipality]
                out_path = make_output_path(department, municipality)

                done_zones = set()

                if os.path.exists(out_path):
                    with open(out_path, "rb") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                done_zones.add(orjson.loads(line)["zone"])

                zones = muni_df["Zone"].unique()
                for zone in tqdm(zones, desc="    Zones", unit="zone", leave=False):
                    if zone in done_zones:
                        continue
                    zone_df = muni_df[muni_df["Zone"] == zone]
                    zone_results = {}
                    locations = zone_df["Location"].unique()

                    for location in tqdm(locations, desc="      Locations",
                                         unit="loc", leave=False):
                        listing_file = zone_df[
                            zone_df["Location"] == location
                        ].iloc[0]["listing_file"]


                        listing_url = (
                            f"{BASE_URL}/{FILE_INIT}"
                            f"{department}/{municipality}/{zone}/{location}/mesas/"
                            f"{listing_file}"
                        )

                        try:
                            session = get_session()
                            resp = session.get(
                                listing_url, timeout=REQUEST_TIMEOUT)
                            resp.raise_for_status()
                            tables_data = orjson.loads(resp.content)
                        except Exception as exc:
                            error_file.write(
                                f"[listing] {listing_url}: {exc}\n"
                            )
                            continue

                        location_results = {}
                        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

                            future_to_number = {
                                executor.submit(
                                    process_table, table["nombre_archivo"]
                                ): str(table["numero"])
                                for table in tables_data
                            }

                            T_progress = tqdm(
                                total=len(future_to_number),
                                desc="        Tables",
                                unit="table",
                                leave=False,
                            )
                            for future in as_completed(future_to_number):
                                table_no = future_to_number[future]
                                result = future.result()
                                T_progress.update(1)

                                if result["success"]:
                                    location_results[table_no] = result["votes"]
                                else:
                                    error_file.write(
                                        f"[table] {result['filename']}: {result['error']}\n"
                                    )
                            T_progress.close()

                        zone_results[location] = location_results

                    append_zone_to_file(out_path, zone, zone_results)


def test_class():
    test = "https://escrutiniospresidente2026.registraduria.gov.co/docs/E14/01/001/08/06/E14_PRE_01_001_008_04_06_002_5077.pdf"
    response = requests.get(test, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    pdf_bytes = response.content
    extractor = E14Extractor(pdf_bytes, canvass_type="V", verbose=True, render_scale=3)
    n, t, c = extractor.resolve_pages()
    extractor.display(t)

if __name__ == "__main__":

    test_class()