"""
Spark DataFrame/SQL pipeline: a fourth comparison point, testing a specific
hypothesis raised by the RDD results -- that PySpark RDDs' per-element Python/JVM
serialization overhead (not Spark itself) was why RDD queries didn't match
DuckDB's near-flat scaling. The DataFrame/SQL API gets Catalyst-optimized,
vectorized execution the same general way DuckDB does, so if that hypothesis is
right, this pipeline's query cost should scale much more like the lakehouse's
than like the RDD pipeline's.

Deliberately uses ONLY native Spark SQL functions (from_json, regexp_replace,
window functions, explode, array_sum) -- no Python UDFs -- since a UDF would
just reintroduce the same per-row Python callback overhead RDDs have, and
wouldn't isolate the variable this comparison is meant to test.
"""

import time

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField

VALID_ACTIONS_SCHEMA = ArrayType(StructType([
    StructField("type", IntegerType()),
    StructField("tiles", ArrayType(IntegerType())),
    StructField("who", ArrayType(IntegerType())),
]))
INT_LIST_SCHEMA = ArrayType(IntegerType())


def spark_df_ingest(spark, csv_path: str):
    """Parse the CSV into a DataFrame with properly-typed nested columns, and
    cache it in memory (forced via .count(), since Spark is lazy)."""
    t0 = time.time()
    raw_df = spark.read.option("header", True).csv(csv_path)
    t_read_schema = time.time() - t0

    t0 = time.time()
    df = raw_df.select(
        F.col("`Data.remain_tiles`").cast("int").alias("remain_tiles"),
        F.from_json(F.col("`Data.hand_tiles`"), INT_LIST_SCHEMA).alias("hand_tiles"),
        F.from_json(F.col("`Data.dora_indicators`"), INT_LIST_SCHEMA).alias("dora_indicators"),
        F.from_json(F.regexp_replace(F.col("`Data.valid_actions`"), "'", '"'), VALID_ACTIONS_SCHEMA).alias("valid_actions"),
        *[F.from_json(F.col(f"`Data.{s}.discards`"), INT_LIST_SCHEMA).alias(f"seat{s}_discards") for s in range(4)],
        *[F.from_json(F.col(f"`Data.{s}.tsumo_giri`"), INT_LIST_SCHEMA).alias(f"seat{s}_tsumo_giri") for s in range(4)],
        *[F.from_json(F.regexp_replace(F.col(f"`Data.{s}.melds`"), "'", '"'), VALID_ACTIONS_SCHEMA).alias(f"seat{s}_melds") for s in range(4)],
    ).cache()
    row_count = df.count()  # force evaluation
    t_parse_cache = time.time() - t0

    total = t_read_schema + t_parse_cache
    return df, {
        "read_and_infer_schema": t_read_schema, "parse_to_typed_and_cache": t_parse_cache,
        "total": total, "row_count": row_count,
    }


def spark_df_query(spark, df) -> dict:
    """Repeated-query cost: round segmentation via a native window function
    (the DataFrame API has these, unlike raw RDDs) + the same aggregation as
    the other three pipelines, all as native (non-UDF) Spark SQL."""
    t0 = time.time()
    w = Window.orderBy(F.monotonically_increasing_id())
    with_round = df.withColumn(
        "is_new_round",
        F.when(F.col("remain_tiles") > F.lag("remain_tiles").over(w), 1).otherwise(0),
    ).withColumn(
        "round_id",
        F.sum("is_new_round").over(w.rowsBetween(Window.unboundedPreceding, 0)),
    )
    n_rounds = with_round.agg(F.max("round_id")).collect()[0][0] + 1  # forces evaluation
    t_roundseg = time.time() - t0

    t0 = time.time()
    agg_exprs = []
    for s in range(4):
        agg_exprs.append(F.sum(F.aggregate(f"seat{s}_tsumo_giri", F.lit(0), lambda acc, x: acc + x)).alias(f"seat{s}_sum"))
        agg_exprs.append(F.sum(F.size(f"seat{s}_tsumo_giri")).alias(f"seat{s}_len"))
    tsumogiri_row = df.agg(*agg_exprs).collect()[0]
    tsumogiri_rates = {
        s: (tsumogiri_row[f"seat{s}_sum"] / tsumogiri_row[f"seat{s}_len"]) if tsumogiri_row[f"seat{s}_len"] else 0.0
        for s in range(4)
    }

    type_counts_rows = (
        df.select(F.explode("valid_actions").alias("action"))
        .groupBy(F.col("action.type").alias("action_type"))
        .count()
        .collect()
    )
    type_counts = {row["action_type"]: row["count"] for row in type_counts_rows}
    t_agg = time.time() - t0

    total = t_roundseg + t_agg
    return {
        "round_seg": t_roundseg, "aggregate": t_agg, "total": total,
        "n_rounds": n_rounds, "tsumogiri_rates": tsumogiri_rates, "type_counts": type_counts,
    }
