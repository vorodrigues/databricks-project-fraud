# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Feature engineering / On-demand
# MAGIC
# MAGIC Tarefa `feature_ondemand` do job `fraud_training`. Garante, de forma **idempotente**, as
# MAGIC **features on-demand** — computadas em tempo de treino/inferência a partir do payload da requisição
# MAGIC (`RequestSource`), sem materialização:
# MAGIC
# MAGIC | Feature | Função | Entrada do payload |
# MAGIC |---|---|---|
# MAGIC | `amount` | `ColumnSelection` — expõe o campo como feature | `amount` (DOUBLE) |
# MAGIC | `ondemand_day_of_week` | `CustomUDF` — UDF UC `day_of_week` | `created_at_brt` (TIMESTAMP) |
# MAGIC
# MAGIC Features já registradas são reutilizadas (`get_feature`). Independe da tarefa `feature_batch`
# MAGIC (as duas rodam em paralelo). Dependências via ambiente serverless da tarefa
# MAGIC (`databricks-feature-engineering>=0.16.0`).

# COMMAND ----------

# DBTITLE 1,Parâmetros
# Parâmetros do job chegam como widgets (valores do target do bundle)
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")

def fqn(name): return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.entities import (
    RequestSource, CustomUDF, ColumnSelection, FieldDefinition, ScalarDataType)

fe = FeatureEngineeringClient()

def ensure_feature(name, **kw):
    try:
        return fe.get_feature(full_name=fqn(name))  # já registrada: reutiliza
    except Exception:
        return fe.create_feature(catalog_name=CATALOG, schema_name=SCHEMA, name=name, **kw)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. `amount` — campo do payload como feature
# MAGIC Com `ColumnSelection` sobre `RequestSource`, o **nome da feature deve ser igual ao da coluna**.

# COMMAND ----------

ensure_feature(
    "amount",
    source=RequestSource(schema=[FieldDefinition(name="amount", data_type=ScalarDataType.DOUBLE)]),
    function=ColumnSelection("amount"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. `ondemand_day_of_week` — UDF UC sobre o timestamp do payload
# MAGIC A função UC Python `day_of_week` é registrada se ainda não existir (`CREATE FUNCTION IF NOT EXISTS`).
# MAGIC `input_bindings` liga o parâmetro da UDF (`dt_transaction`) ao campo `created_at_brt` da requisição.

# COMMAND ----------

spark.sql(f"""
CREATE FUNCTION IF NOT EXISTS {CATALOG}.{SCHEMA}.day_of_week(dt_transaction TIMESTAMP)
RETURNS STRING LANGUAGE PYTHON
COMMENT 'Returns the full English day-of-week name for a given transaction timestamp.'
AS $$
return dt_transaction.strftime("%A")
$$
""")

# COMMAND ----------

ensure_feature(
    "ondemand_day_of_week",
    source=RequestSource(schema=[FieldDefinition(name="created_at_brt", data_type=ScalarDataType.TIMESTAMP)]),
    function=CustomUDF(
        function_name=fqn("day_of_week"),
        input_bindings={"dt_transaction": "created_at_brt"}))
