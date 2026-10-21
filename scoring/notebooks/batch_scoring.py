# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Predição em batch com `fraud_model@champion`
# MAGIC
# MAGIC Tarefa `score` do job `fraud_batch_scoring` (agendado diariamente, inativo por padrão):
# MAGIC
# MAGIC 1. Lê a tabela fonte indicada pelo parâmetro `source_table` (contrato: `holder_id`, `created_at_brt`, `amount`).
# MAGIC 2. Resolve a versão apontada pelo alias `champion` e pontua com `fe.score_batch` — as features
# MAGIC    (Feature Views batch e on-demand) são calculadas a partir
# MAGIC    do **feature spec** salvo com o modelo,
# MAGIC    com correção point-in-time, igual ao treino.
# MAGIC 3. Grava as predições em `fraud_batch_predictions` do target em modo **append**, com a versão do modelo,
# MAGIC    a tabela fonte e o instante do scoring.

# COMMAND ----------

# DBTITLE 1,Parâmetros do job
# source_table: nome totalmente qualificado (catalog.schema.tabela); o job param de mesmo nome alimenta o widget
dbutils.widgets.text("source_table", "")
dbutils.widgets.text("model_name", "")  # registered model pontuado (catalog.schema.nome)
dbutils.widgets.text("catalog", "")  # catálogo/schema do target: destino das predições
dbutils.widgets.text("schema", "")
SOURCE_TABLE = dbutils.widgets.get("source_table")

MODEL_NAME = dbutils.widgets.get("model_name")
ALIAS = "champion"
OUTPUT_TABLE = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}.fraud_batch_predictions"
KEYS = ["holder_id", "created_at_brt", "amount"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tabela de saída
# MAGIC Criada uma única vez (`IF NOT EXISTS`); cada execução acrescenta linhas.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
  holder_id      STRING    COMMENT 'Titular da transação',
  created_at_brt TIMESTAMP COMMENT 'Instante da transação (BRT)',
  amount         DOUBLE    COMMENT 'Valor da transação',
  prediction     DOUBLE    COMMENT 'Predição do modelo (1 = fraude)',
  model_version  INT       COMMENT 'Versão de fraud_model usada (alias champion)',
  source_table   STRING    COMMENT 'Tabela fonte pontuada',
  scored_at      TIMESTAMP COMMENT 'Instante do scoring'
)
COMMENT 'Predições em batch de fraud_model@champion (job fraud_batch_scoring)'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Scoring com `score_batch`
# MAGIC O alias é resolvido para um número de versão antes do scoring: a versão usada fica registrada
# MAGIC na saída mesmo que `champion` mude durante a execução. Só as colunas do contrato da fonte são enviadas;
# MAGIC o restante das entradas vem do feature spec.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
version = int(MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version)

fe = FeatureEngineeringClient()
source_df = spark.table(SOURCE_TABLE).select(*KEYS)
scored = fe.score_batch(model_uri=f"models:/{MODEL_NAME}/{version}", df=source_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gravação (append)

# COMMAND ----------

(scored.select(
    *KEYS,
    F.col("prediction").cast("double").alias("prediction"),
    F.lit(version).alias("model_version"),
    F.lit(SOURCE_TABLE).alias("source_table"),
    F.current_timestamp().alias("scored_at"))
 .write.mode("append").saveAsTable(OUTPUT_TABLE))
