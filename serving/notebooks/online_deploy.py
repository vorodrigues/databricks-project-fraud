# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Model Serving do modelo antifraude PIX (route-optimized + inference table)
# MAGIC
# MAGIC Tarefa `deploy` do job `fraud_online_deploy`: publica a versão `champion` de `fraud_model` do target
# MAGIC (registrado com o **feature spec**) em um **endpoint de Model Serving**. O endpoint resolve
# MAGIC automaticamente, a cada request, as **6 features batch** (lookup online no Lakebase compartilhado) e a
# MAGIC on-demand `amount` (via `RequestSource`) — o cliente envia apenas `holder_id` + `created_at_brt` + `amount`.
# MAGIC
# MAGIC ## O que este notebook faz (idempotente)
# MAGIC 1. **Deploy via Databricks SDK** — `create_and_wait` na 1a vez; `update_config_and_wait` +
# MAGIC    `put_ai_gateway` se o endpoint ja existir (nunca recria destrutivamente).
# MAGIC 2. **Route-optimization** (`route_optimized=True`) — rota de baixa latencia; definida **apenas na
# MAGIC    criacao**. Exige query via `endpoint_url` (data plane) com token OAuth.
# MAGIC 3. **Inference table via AI Gateway** — `AiGatewayInferenceTableConfig` grava os payloads no
# MAGIC    catalogo/schema do target (substitui o `auto_capture_config` legado).
# MAGIC 4. **Permissao de consulta** — concede `CAN_QUERY` no endpoint ao grupo de leitores (aditivo).
# MAGIC
# MAGIC > O `databricks-feature-lookup` vem nativo no Model Serving — nao precisa empacotar.

# COMMAND ----------

# DBTITLE 1,Parametros do endpoint
# valores vem dos parametros do job (bundle); vazios em execucao isolada
dbutils.widgets.text("endpoint_name", "")   # nome do endpoint (<prefix>_fraud_serving)
dbutils.widgets.text("model_name", "")      # modelo UC empacotado com feature spec (catalog.schema.nome)
dbutils.widgets.text("catalog", "")         # catalogo/schema do target: destino da inference table
dbutils.widgets.text("schema", "")
dbutils.widgets.text("viewers_group", "")   # grupo que recebe CAN_QUERY no endpoint

CATALOG        = dbutils.widgets.get("catalog")
PROJECT_SCHEMA = dbutils.widgets.get("schema")
ENDPOINT_NAME  = dbutils.widgets.get("endpoint_name")
VIEWERS_GROUP  = dbutils.widgets.get("viewers_group")
SERVED_NAME    = "fraud-model"                     # nome da served entity dentro do endpoint
INFERENCE_TABLE_PREFIX = "fraud_serving_inference"   # AI Gateway -> <cat>.<schema>.<prefix>_payload
ALIAS          = "champion"

MODEL_FULL_NAME = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md ## 1. Resolver a versao `champion` do modelo
# MAGIC O alias e resolvido para um numero de versao: o endpoint serve uma versao fixa, e um novo deploy
# MAGIC acompanha o alias apos cada retreino.

# COMMAND ----------

from mlflow.tracking import MlflowClient

mlflow_client = MlflowClient(registry_uri="databricks-uc")
model_version = mlflow_client.get_model_version_by_alias(MODEL_FULL_NAME, ALIAS).version
print(f"modelo={MODEL_FULL_NAME} | versao {ALIAS}={model_version}")

# COMMAND ----------

# MAGIC %md ## 2. Config do endpoint (served entity + traffic + AI Gateway inference table)
# MAGIC - **served entity**: modelo UC na versao `champion`, workload CPU/Small, scale-to-zero.
# MAGIC - **AI Gateway inference table**: grava os payloads de request/response no UC do target
# MAGIC   (`<catalog>.<schema>.<prefix>_payload`).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, TrafficConfig, Route,
    ServingModelWorkloadType, AiGatewayConfig, AiGatewayInferenceTableConfig,
)

w = WorkspaceClient()

served_entity = ServedEntityInput(
    name=SERVED_NAME,
    entity_name=MODEL_FULL_NAME,
    entity_version=model_version,
    workload_type=ServingModelWorkloadType.CPU,
    workload_size="Small",
    scale_to_zero_enabled=True,
)
core_config = EndpointCoreConfigInput(
    name=ENDPOINT_NAME,
    served_entities=[served_entity],
    traffic_config=TrafficConfig(routes=[Route(served_model_name=SERVED_NAME, traffic_percentage=100)]),
)
ai_gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        enabled=True,
        catalog_name=CATALOG,
        schema_name=PROJECT_SCHEMA,
        table_name_prefix=INFERENCE_TABLE_PREFIX,
    )
)

# COMMAND ----------

# MAGIC %md ## 3. Deploy idempotente
# MAGIC `create_and_wait` (com `route_optimized` + `ai_gateway`) apenas na 1a vez. Se ja existe, atualiza a
# MAGIC config servida (`update_config_and_wait`) e a inference table (`put_ai_gateway`) — sem recriar.
# MAGIC `route_optimized` so pode ser definido na criacao (nao muda via update).

# COMMAND ----------

from databricks.sdk.errors import ResourceDoesNotExist

try:
    existing = w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"endpoint '{ENDPOINT_NAME}' existe (route_optimized={existing.route_optimized}) -> update")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[served_entity],
        traffic_config=core_config.traffic_config,
    )
    w.serving_endpoints.put_ai_gateway(
        name=ENDPOINT_NAME,
        inference_table_config=ai_gateway.inference_table_config,
    )
except ResourceDoesNotExist:
    print(f"endpoint '{ENDPOINT_NAME}' nao existe -> create (route_optimized=True)")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=core_config,
        route_optimized=True,
        ai_gateway=ai_gateway,
    )

ep = w.serving_endpoints.get(ENDPOINT_NAME)
print(f"estado={ep.state.ready} | config_update={ep.state.config_update} | route_optimized={ep.route_optimized}")
print(f"endpoint_url={ep.endpoint_url}")
print(f"ai_gateway.inference_table={ep.ai_gateway.inference_table_config if ep.ai_gateway else None}")

# COMMAND ----------

# MAGIC %md ## 4. Permissao de consulta para o grupo de leitores
# MAGIC `update_permissions` **acrescenta** `CAN_QUERY` ao grupo, preservando as demais permissoes do endpoint
# MAGIC (`set_permissions` substituiria a ACL inteira).

# COMMAND ----------

from databricks.sdk.service.serving import ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel

acl = w.serving_endpoints.update_permissions(
    serving_endpoint_id=ep.id,
    access_control_list=[ServingEndpointAccessControlRequest(
        group_name=VIEWERS_GROUP, permission_level=ServingEndpointPermissionLevel.CAN_QUERY)])
for a in acl.access_control_list:
    print(a.user_name or a.group_name or a.service_principal_name, [p.permission_level.value for p in a.all_permissions])

# COMMAND ----------

# DBTITLE 1,Evidencia estruturada
import json

ep = w.serving_endpoints.get(ENDPOINT_NAME)
itc = ep.ai_gateway.inference_table_config if ep.ai_gateway else None
dbutils.notebook.exit(json.dumps({
    "endpoint_name":       ENDPOINT_NAME,
    "endpoint_id":         ep.id,
    "state_ready":         str(ep.state.ready),
    "config_update":       str(ep.state.config_update),
    "route_optimized":     ep.route_optimized,
    "endpoint_url":        ep.endpoint_url,
    "model_full_name":     MODEL_FULL_NAME,
    "model_version":       model_version,
    "inference_table":     f"{CATALOG}.{PROJECT_SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload",
    "inference_enabled":   (itc.enabled if itc else None),
}))
