# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — online feature store (Lakebase)
# MAGIC
# MAGIC O online store é uma instância Lakebase **compartilhada** por todos os targets: as features batch são materializadas nela e lidas pelo Model Serving em tempo de inferência.
# MAGIC
# MAGIC Este notebook:
# MAGIC 1. reutiliza a instância existente ou a cria se ausente (nunca recria nem remove);
# MAGIC 2. garante `CAN_USE` na instância às identidades de execução de todos os targets (necessário para materializar features online).
# MAGIC
# MAGIC Referência: [Online feature store](https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store)

# COMMAND ----------

dbutils.widgets.text("online_store", "")
dbutils.widgets.text("principals", "")  # identidades separadas por vírgula (e-mail de usuário ou application id de SP)

ONLINE_STORE = dbutils.widgets.get("online_store")
PRINCIPALS = [p.strip() for p in dbutils.widgets.get("principals").split(",") if p.strip()]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instância
# MAGIC
# MAGIC `get_online_store` → `create_online_store` só se ausente; aguarda o estado `AVAILABLE`.

# COMMAND ----------

import time
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

store = fe.get_online_store(name=ONLINE_STORE)
if store is None:
    store = fe.create_online_store(name=ONLINE_STORE, capacity="CU_1")
while "AVAILABLE" not in str(store.state):
    if any(s in str(store.state) for s in ("FAILED", "DELETING")):
        raise RuntimeError(f"Online store {ONLINE_STORE}: {store.state}")
    time.sleep(20)
    store = fe.get_online_store(name=ONLINE_STORE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Acesso das identidades
# MAGIC
# MAGIC Permissões da instância (`database-instances`): concede `CAN_USE` apenas às identidades que ainda não têm `CAN_USE` ou `CAN_MANAGE` (atualização aditiva).
# MAGIC
# MAGIC Ler e alterar a ACL exige `CAN_MANAGE`. Uma identidade com apenas `CAN_USE` (que já acessa a instância acima) não consegue verificá-la: a concessão às demais identidades fica com quem gerencia a instância.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import PermissionDenied
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel

w = WorkspaceClient()
OBJECT_TYPE = "database-instances"

try:
    acl = w.permissions.get(OBJECT_TYPE, ONLINE_STORE).access_control_list or []
except PermissionDenied:
    acl = None

if acl is None:
    missing = []
    print(f"Sem CAN_MANAGE em {ONLINE_STORE}: ACL não verificada para {PRINCIPALS}")
else:
    granted = {
        a.user_name or a.service_principal_name
        for a in acl
        if any(p.permission_level in (PermissionLevel.CAN_USE, PermissionLevel.CAN_MANAGE) for p in a.all_permissions or [])
    }
    missing = [p for p in PRINCIPALS if p not in granted]
if missing:
    w.permissions.update(OBJECT_TYPE, ONLINE_STORE, access_control_list=[
        AccessControlRequest(user_name=p, permission_level=PermissionLevel.CAN_USE) if "@" in p
        else AccessControlRequest(service_principal_name=p, permission_level=PermissionLevel.CAN_USE)
        for p in missing
    ])

dbutils.notebook.exit(f"{store.name} {store.state} granted={missing} acl_checked={acl is not None}")
