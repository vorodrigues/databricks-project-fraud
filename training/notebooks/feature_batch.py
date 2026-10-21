# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Feature engineering / Batch
# MAGIC
# MAGIC Tarefa `feature_batch` do job `fraud_training`. Garante, de forma **idempotente**, as
# MAGIC **Feature Views declarativas batch** e suas **materializações gerenciadas** (offline Delta + online
# MAGIC Lakebase), terminando com a tabela offline populada para o treino.
# MAGIC
# MAGIC ## Fluxo
# MAGIC 1. **Features declarativas** (`fe.create_feature`) sobre `DeltaTableSource`: soma e contagem de
# MAGIC    `amount` por `holder_id` em janelas de 1d, 7d e 30d. Reutiliza as já registradas (`get_feature`).
# MAGIC 2. **Materialização** (`fe.materialize_features`): pipeline gerenciado que grava o offline store e
# MAGIC    sincroniza o online store (`TableTrigger` atualiza a cada mudança da fonte). Só é criada para
# MAGIC    features ainda sem materialização — reexecutar não duplica pipelines.
# MAGIC 3. **Espera** a tabela offline ter dados (backfill assíncrono na primeira materialização).
# MAGIC
# MAGIC Dependências via ambiente serverless da tarefa (`databricks-feature-engineering>=0.16.0`).
# MAGIC Pré-requisitos (garantidos pelo job `fraud_setup`): tabela fonte `raw_pix_transactions` e
# MAGIC online store compartilhada entre os targets (`online_store`).

# COMMAND ----------

# DBTITLE 1,Parâmetros
import time
from datetime import timedelta

# Parâmetros do job chegam como widgets (valores do target do bundle)
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("online_store", "")

CATALOG        = dbutils.widgets.get("catalog")
SCHEMA         = dbutils.widgets.get("schema")
RAW_TABLE      = "raw_pix_transactions"        # fonte de movimentos crus (evento a evento)
ONLINE_STORE   = dbutils.widgets.get("online_store")  # online store Lakebase existente
OFFLINE_PREFIX = "batch_pix_features"          # prefixo da tabela offline gerenciada
ONLINE_PREFIX  = "batch_pix_features_online"   # prefixo da tabela online gerenciada

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Feature Views declarativas — `batch_pix_{amount,transfers}_{1,7,30}d`
# MAGIC `AggregationFunction(Op(input="amount"), janela)` sobre a fonte Delta, com `holder_id` como entidade e
# MAGIC `created_at_brt` como coluna temporal (point-in-time). 1d usa `TumblingWindow` (`SlidingWindow` exige
# MAGIC slide < janela); 7d/30d usam `SlidingWindow` com slide diário.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.entities import (
    DeltaTableSource, AggregationFunction, Sum, Count,
    SlidingWindow, TumblingWindow, OfflineStoreConfig, OnlineStoreConfig, TableTrigger)

FEATURE_DEFS = [("batch_pix_amount", Sum), ("batch_pix_transfers", Count)]  # (prefixo, operador)
WINDOWS = [1, 7, 30]                                                        # dias

fe = FeatureEngineeringClient()
source = DeltaTableSource(catalog_name=CATALOG, schema_name=SCHEMA, table_name=RAW_TABLE)

def window(days):
    if days == 1:
        return TumblingWindow(window_duration=timedelta(days=1))
    return SlidingWindow(window_duration=timedelta(days=days), slide_duration=timedelta(days=1))

def ensure_feature(name, Op, days):
    try:
        return fe.get_feature(full_name=f"{CATALOG}.{SCHEMA}.{name}")  # já registrada: reutiliza
    except Exception:
        return fe.create_feature(
            source=source,
            function=AggregationFunction(Op(input="amount"), window(days)),
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            entity=["holder_id"],
            timeseries_column="created_at_brt",
            name=name)

features = [ensure_feature(f"{prefix}_{w}d", Op, w) for prefix, Op in FEATURE_DEFS for w in WINDOWS]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Materialização gerenciada — offline (Delta) + online (Lakebase)
# MAGIC `fe.list_materialized_features` lista as materializações de cada feature; `materialize_features` só é
# MAGIC chamado para as que não têm nenhuma (chamá-lo de novo criaria um pipeline duplicado). Os nomes das
# MAGIC tabelas são gerados (prefixo + sufixo aleatório) e por isso sempre descobertos pela API.

# COMMAND ----------

missing = [f for f in features if not fe.list_materialized_features(feature_name=f.full_name)]
if missing:
    fe.materialize_features(
        features=missing,
        offline_config=OfflineStoreConfig(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            table_name_prefix=OFFLINE_PREFIX),
        online_config=OnlineStoreConfig(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            table_name_prefix=ONLINE_PREFIX,
            online_store_name=ONLINE_STORE),
        trigger=TableTrigger())

offline_tables = sorted({m.table_name for f in features
                         for m in fe.list_materialized_features(feature_name=f.full_name) if not m.is_online})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tabela offline populada
# MAGIC Na primeira materialização o backfill é assíncrono (~10-20 min): aguarda até a tabela offline ter dados
# MAGIC e falha a tarefa se isso não ocorrer dentro do limite (o retry do job reexecuta de forma idempotente).

# COMMAND ----------

def populated(table):
    try:
        return spark.table(table).limit(1).count() > 0
    except Exception:  # tabela ainda não criada pelo pipeline
        return False

deadline = time.time() + 30 * 60
while not all(populated(t) for t in offline_tables):
    if time.time() > deadline:
        raise TimeoutError(f"Tabela offline sem dados após 30 min: {offline_tables}")
    time.sleep(30)
