
BASE_URL = "https://escrutiniospresidente2026.registraduria.gov.co"

LOCS_JSON = "src/files_loc.json"

FILE_INIT = "data/esc/v1/actas-documentos/001/"

OUTPUT_DIR = "data"

ERROR_LOG_FILE = "error_logs.json"

MAX_WORKERS = 8

REQUEST_TIMEOUT = (10, 120)

RETRIES = 3

BACKOFF_FACTOR = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

thread_local = threading.local()

def get_session():

    if not hasattr(thread_local, "session"):

        session = requests.Session()

        retry = Retry(
            total=RETRIES,
            connect=RETRIES,
            read=RETRIES,
            status=RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )

        adapter = HTTPAdapter(
            pool_connections=MAX_WORKERS,
            pool_maxsize=MAX_WORKERS,
            max_retries=retry,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        thread_local.session = session

    return thread_local.session

def store_json(data, filename):

    path = os.path.join(
        OUTPUT_DIR,
        f"{filename}.json"
    )

    with open(path, "wb") as f:

        f.write(
            orjson.dumps(
                data,
                option=orjson.OPT_INDENT_2
            )
        )


def parse_location(dir_path):

    return (
        dir_path
        .replace(FILE_INIT, "")
        .replace("/mesas/", "")
        .split("/")
    )

def create_municipality_filename(location):

    """
    Generates:

    d60_m001

    WITHOUT zXX or pXX
    """

    return (
        f"d{location[0]}_"
        f"m{location[1]}"
    )


def process_table(task):

    """
    Complete pipeline for ONE table:
    - download pdf
    - run VoteExtractor
    - cleanup temp file
    """

    table_data, zone_key, position_key = task

    mesa_num = table_data["numero"]

    pdf_url = BASE_URL + table_data["nombre_archivo"]

    tmp_path = None

    try:

        session = get_session()

        with session.get(
            pdf_url,
            stream=True,
            timeout=REQUEST_TIMEOUT
        ) as response:

            response.raise_for_status()

            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False
            ) as tmp:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        tmp.write(chunk)

                tmp_path = tmp.name

        extractor = VoteExtractor(tmp_path)

        votes_json = extractor.get_votes_as_json(
            clean=False
        )

        return {
            "success": True,
            "zone": zone_key,
            "position": position_key,
            "mesa": mesa_num,
            "votes": votes_json
        }

    except Exception as e:

        return {
            "success": False,
            "zone": zone_key,
            "position": position_key,
            "mesa": mesa_num,
            "error": str(e)
        }

    finally:

        if tmp_path and os.path.exists(tmp_path):

            try:
                os.remove(tmp_path)

            except:
                pass

def main():

    with open(
        LOCS_JSON,
        "rb"
    ) as f:

        raw_data = orjson.loads(f.read())

    data = {
        k: v
        for k, v in raw_data.items()
        if k.startswith(FILE_INIT)
    }

    print(f"Found {len(data)} files to process.")

    grouped = defaultdict(lambda: defaultdict(list))

    for dir_path, file_name in data.items():

        location = parse_location(dir_path)

        dept_key            = f"d{location[0]}"
        municipality_filename = create_municipality_filename(location)

        grouped[dept_key][municipality_filename].append(
            (dir_path, file_name, location)
        )

    total_depts   = len(grouped)
    total_munis   = sum(len(munis) for munis in grouped.values())
    total_zones   = sum(
        len({f"z{loc[2]}" for _, _, loc in items})
        for munis in grouped.values()
        for items in munis.values()
    )

    error_logs = []

    dept_bar  = tqdm(total=total_depts,  desc="Department",   position=0, leave=True)
    muni_bar  = tqdm(total=total_munis,  desc="Municipality", position=1, leave=True)
    zone_bar  = tqdm(total=total_zones,  desc="Zone",         position=2, leave=True)

    try:

        for dept_key, municipalities in grouped.items():

            dept_bar.set_description(f"Department {dept_key}")

            for municipality_filename, municipality_items in municipalities.items():

                muni_bar.set_description(f"Municipality {municipality_filename}")

                municipality_json = {}
                all_tasks         = []

                zones_in_muni = set()

                for dir_path, file_name, location in municipality_items:

                    zone_key     = f"z{location[2]}"
                    position_key = location[3]

                    municipality_json.setdefault(zone_key, {}).setdefault(position_key, {})

                    tables_url = f"{BASE_URL}/{dir_path}{file_name}".replace("///", "/").replace("//", "/")
                    tables_url = tables_url.replace("https:/", "https://")

                    try:

                        session  = get_session()
                        response = session.get(
                            tables_url,
                            timeout=REQUEST_TIMEOUT
                        )
                        response.raise_for_status()
                        tables_json = response.json()

                    except Exception as e:

                        error_logs.append({
                            "file":     file_name,
                            "location": location,
                            "error":    f"Tables JSON error: {str(e)}"
                        })
                        continue

                    for table_data in tables_json:
                        all_tasks.append((table_data, zone_key, position_key))

                    zones_in_muni.add(zone_key)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

                    futures = {
                        executor.submit(process_table, task): task
                        for task in all_tasks
                    }

                    completed_zones = set()

                    for future in as_completed(futures):

                        result = future.result()

                        if result["success"]:

                            municipality_json[
                                result["zone"]
                            ][
                                result["position"]
                            ][
                                f"mesa {result['mesa']}"
                            ] = result["votes"]

                        else:

                            error_logs.append({
                                "zone":     result["zone"],
                                "position": result["position"],
                                "mesa":     result["mesa"],
                                "error":    result["error"]
                            })

                        z = result["zone"]
                        if z not in completed_zones and z in zones_in_muni:
                            zone_tasks_done = all(
                                f.done()
                                for f, t in futures.items()
                                if t[1] == z
                            )
                            if zone_tasks_done:
                                completed_zones.add(z)
                                zone_bar.set_description(f"Zone {z}")
                                zone_bar.update(1)

                store_json(municipality_json, municipality_filename)
                muni_bar.update(1)

            dept_bar.update(1)

    finally:

        dept_bar.close()
        muni_bar.close()
        zone_bar.close()

    with open(ERROR_LOG_FILE, "wb") as f:
        f.write(
            orjson.dumps(
                error_logs,
                option=orjson.OPT_INDENT_2
            )
        )

    print("\nDONE")


if __name__ == "__main__":

    main()