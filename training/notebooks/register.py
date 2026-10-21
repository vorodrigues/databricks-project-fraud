# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Registro no UC, tags e alias `champion`
# MAGIC
# MAGIC Tarefa `register` do job `fraud_training` (só executa no ramo `true` do `quality_gate`):
# MAGIC
# MAGIC 1. Lê o **task value** `model_uri` publicado por `train` e registra uma nova **versão** de
# MAGIC    `fraud_model` do target no Unity Catalog.
# MAGIC 2. Aplica as tags `team` e `retention` (parâmetros do job) à **versão** e ao **registered model**.
# MAGIC 3. Aplica a **tag governada** `confidentiality = confidential` ao registered model — a chave tem
# MAGIC    tag policy (job `fraud_setup`), então o UC valida o valor contra a lista permitida.
# MAGIC 4. Aponta o alias `champion` para a nova versão (o endpoint de serving não é alterado aqui).
# MAGIC
# MAGIC Em registered models e versões, tags UC são aplicadas pela API MLflow
# MAGIC (`set_registered_model_tag` / `set_model_version_tag`) — não há SQL `SET TAG` para modelos.
# MAGIC Reexecutável (Repair run): se a run de origem já tem versão registrada, ela é reaproveitada.

# COMMAND ----------

# DBTITLE 1,Parâmetros do job
# team/retention/model_name vêm dos parâmetros do job; model_uri só é preenchido em execução isolada
dbutils.widgets.text("team", "fraud")
dbutils.widgets.text("retention", "365d")
dbutils.widgets.text("model_uri", "")  # ex.: runs:/<run_id>/model; vazio → task value de `train`
dbutils.widgets.text("model_name", "")  # registered model de destino (catalog.schema.nome)

TAGS = {"team": dbutils.widgets.get("team"), "retention": dbutils.widgets.get("retention")}
GOVERNED_TAGS = {"confidentiality": "confidential"}

MODEL_NAME = dbutils.widgets.get("model_name")
ALIAS = "champion"

model_uri = dbutils.widgets.get("model_uri") or dbutils.jobs.taskValues.get(taskKey="train", key="model_uri")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Nova versão a partir do modelo logado por `train`
# MAGIC `mlflow.register_model` cria a versão no UC preservando o feature spec logado por `fe.log_model`.
# MAGIC Para idempotência, uma versão já registrada a partir da mesma run é reaproveitada.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

run_id = model_uri.split("/")[1]  # runs:/<run_id>/model
existing = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'") if v.run_id == run_id]
version = existing[0].version if existing else mlflow.register_model(model_uri, MODEL_NAME).version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tags UC na versão e no registered model
# MAGIC `team`/`retention` na versão criada e no modelo; `confidentiality` (governada) só no modelo.

# COMMAND ----------

for key, value in TAGS.items():
    client.set_model_version_tag(MODEL_NAME, version, key, value)

for key, value in {**TAGS, **GOVERNED_TAGS}.items():
    client.set_registered_model_tag(MODEL_NAME, key, value)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Promoção: alias `champion` → nova versão
# MAGIC Quem consome `fraud_model@champion` (ex.: o job de predição em batch) passa a usar esta versão.

# COMMAND ----------

client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
dbutils.notebook.exit(str(version))
