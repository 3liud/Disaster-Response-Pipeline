from dagster import job, op

from src.ingestion.fetch_reliefweb import fetch_reliefweb, _write_csv, _read_cursor


@op
def fetch_to_csv():
    since = _read_cursor()
    rows = list(fetch_reliefweb(since_iso=since))
    path = _write_csv(rows)
    return str(path)


@job
def ingest_reliefweb_job():
    fetch_to_csv()
