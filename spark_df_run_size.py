import json
import os
import sys
import time

from pyspark.sql import SparkSession
from benchmark_spark_df import spark_df_ingest, spark_df_query

RESULTS_PATH = "/home/claude/lakehouse_study/results_spark_df.json"


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    size = int(sys.argv[1])
    results = load_results()
    key = str(size)

    csv_path = f"/home/claude/lakehouse_study/slice_{size}.csv"

    print(f"=== size {size} ===")

    t0 = time.time()
    spark = (
        SparkSession.builder.master("local[1]").appName(f"df_bench_{size}")
        .config("spark.driver.memory", "2g").config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    t_startup = time.time() - t0
    print(f"session startup: {t_startup:.3f}s")

    df, ing = spark_df_ingest(spark, csv_path)
    print("ingest:", {k: round(v, 4) if isinstance(v, float) else v for k, v in ing.items()})

    q = spark_df_query(spark, df)
    print("query:", {k: round(v, 4) for k, v in q.items() if not isinstance(v, dict)})

    spark.stop()

    results[key] = {
        "session_startup": t_startup,
        "ingest": ing,
        "query": {k: v for k, v in q.items() if not isinstance(v, dict)},
    }
    save_results(results)
    print(f"saved results for size {size}")
