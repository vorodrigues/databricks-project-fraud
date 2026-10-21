# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — dados sintéticos
# MAGIC
# MAGIC Gera, no schema do target, as duas tabelas de origem do modelo antifraude PIX:
# MAGIC
# MAGIC | Tabela | Conteúdo |
# MAGIC |---|---|
# MAGIC | `raw_pix_transactions` | ~1M transações PIX de ~10 mil titulares nos 45 dias anteriores à execução (fonte das Feature Views batch) |
# MAGIC | `raw_fraud_labels` | ~100 mil transações amostradas dos últimos 7 dias, uma por âncora (`holder_id`, `created_at_brt`), com ~5% de fraude |
# MAGIC
# MAGIC A geração é determinística (semente fixa) e idempotente: tabelas já existentes não são regeneradas.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TRANSACTIONS = f"{CATALOG}.{SCHEMA}.raw_pix_transactions"
LABELS = f"{CATALOG}.{SCHEMA}.raw_fraud_labels"

SEED = 42
N_HOLDERS = 10_000
N_TRANSACTIONS = 1_000_000
N_LABELS = 100_000
DAYS = 45          # período das transações
ANCHOR_DAYS = 7    # âncoras nos últimos dias do período (janela de 30d sempre completa)
FRAUD_RATE = 0.05

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transações PIX
# MAGIC
# MAGIC - `holder_id`: `H` + 10 dígitos; a atividade é concentrada em parte dos titulares (distribuição assimétrica).
# MAGIC - `created_at_brt`: instante uniforme nos 45 dias anteriores à data da execução (horário de Brasília).
# MAGIC - `amount`: valor log-normal (mediana ~R$ 90).

# COMMAND ----------

from datetime import datetime, timedelta, timezone
from pyspark.sql import functions as F, Window

if not spark.catalog.tableExists(TRANSACTIONS):
    today_brt = datetime.now(timezone(timedelta(hours=-3))).date()
    start = F.lit(str(today_brt - timedelta(days=DAYS))).cast("timestamp")
    holder_idx = (F.pow(F.rand(SEED), 1.5) * N_HOLDERS).cast("long")
    transactions = (
        spark.range(0, N_TRANSACTIONS, numPartitions=8)
        .select(
            F.format_string("H%010d", F.pmod(F.xxhash64(holder_idx), F.lit(10**10))).alias("holder_id"),
            F.timestamp_seconds(F.unix_timestamp(start) + (F.rand(SEED + 1) * DAYS * 86400).cast("long")).alias("created_at_brt"),
            F.round(F.exp(F.randn(SEED + 2) + 4.5), 2).alias("amount"),
        )
    )
    transactions.write.saveAsTable(TRANSACTIONS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rótulos de fraude
# MAGIC
# MAGIC As âncoras são transações reais de `raw_pix_transactions` dos últimos 7 dias, sem repetição de (`holder_id`, `created_at_brt`).
# MAGIC O `label` tem sinal sintético: o score cresce com o `amount` e com o número de transferências do titular nas 24h anteriores, mais um ruído determinístico; os 5% maiores scores são fraude.

# COMMAND ----------

if not spark.catalog.tableExists(LABELS):
    transactions = spark.table(TRANSACTIONS)
    end = transactions.select(F.date_add(F.to_date(F.max("created_at_brt")), 1)).first()[0]
    anchor_start = F.lit(str(end - timedelta(days=ANCHOR_DAYS))).cast("timestamp")

    last_24h = Window.partitionBy("holder_id").orderBy(F.col("created_at_brt").cast("long")).rangeBetween(-86400, 0)
    noise_u = (F.pmod(F.xxhash64("holder_id", "created_at_brt", F.lit(SEED)), F.lit(1_000_000)) + 0.5) / 1_000_000

    anchors = (
        transactions
        .withColumn("transfers_24h", F.count("*").over(last_24h))
        .where(F.col("created_at_brt") >= anchor_start)
        .dropDuplicates(["holder_id", "created_at_brt"])
        .orderBy(F.xxhash64("holder_id", "created_at_brt"))
        .limit(N_LABELS)
        .withColumn("score", F.log("amount") + F.log("transfers_24h") + 1.5 * F.log(noise_u / (1 - noise_u)))
    )
    threshold = anchors.select(F.percentile_approx("score", 1 - FRAUD_RATE, 10_000)).first()[0]
    labels = anchors.select(
        "holder_id", "created_at_brt", "amount",
        (F.col("score") >= threshold).cast("int").alias("label"),
    )
    labels.write.saveAsTable(LABELS)
