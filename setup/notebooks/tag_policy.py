# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — tags governadas `confidentiality` e `team`
# MAGIC
# MAGIC **Governed tags** são tags com uma *tag policy* definida no nível da conta: a policy fixa a chave e a lista de valores permitidos, e só usuários com permissão `ASSIGN` podem aplicá-la a objetos do Unity Catalog.
# MAGIC
# MAGIC As policies são **compartilhadas** por todos os targets. Este notebook garante:
# MAGIC - `confidentiality` com os valores `public`, `internal`, `confidential` e `restricted`;
# MAGIC - `team` contendo o valor `fraud`;
# MAGIC - a permissão `ASSIGN` às identidades de execução de todos os targets.
# MAGIC
# MAGIC As tags são aplicadas depois pela tarefa `register` do job de treino ao modelo registrado.
# MAGIC
# MAGIC Referência: [Governed tags](https://docs.databricks.com/aws/en/admin/governed-tags/)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definição desejada

# COMMAND ----------

dbutils.widgets.text("principals", "")  # identidades separadas por vírgula (e-mail de usuário ou application id de SP)
PRINCIPALS = [p.strip() for p in dbutils.widgets.get("principals").split(",") if p.strip()]

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.iam import GrantRule, RuleSetUpdateRequest
from databricks.sdk.service.tags import TagPolicy, Value

POLICIES = {
    "confidentiality": ("Classificação de confidencialidade do ativo", ["public", "internal", "confidential", "restricted"]),
    "team": ("Time responsável pelo ativo", ["fraud"]),
}

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criação idempotente
# MAGIC
# MAGIC API `tag_policies` do SDK:
# MAGIC - policy inexistente → `create_tag_policy`;
# MAGIC - policy existente sem algum dos valores → `update_tag_policy` com `update_mask="values"`, acrescentando só os valores faltantes (os existentes são preservados);
# MAGIC - policy já conforme → nenhuma alteração.

# COMMAND ----------

actions = {}
for key, (description, values) in POLICIES.items():
    try:
        current = w.tag_policies.get_tag_policy(key)
    except NotFound:
        current = None

    if current is None:
        w.tag_policies.create_tag_policy(TagPolicy(tag_key=key, description=description, values=[Value(name=v) for v in values]))
        actions[key] = "created"
        continue

    existing = [v.name for v in current.values or []]
    missing = [v for v in values if v not in existing]
    if missing:
        w.tag_policies.update_tag_policy(
            key, TagPolicy(tag_key=key, values=[Value(name=v) for v in existing + missing]), update_mask="values")
        actions[key] = f"added {missing}"
    else:
        actions[key] = "unchanged"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Permissão `ASSIGN`
# MAGIC
# MAGIC As permissões de cada policy ficam no *rule set* da conta `accounts/<account_id>/tagPolicies/<policy_id>/ruleSets/default` (API Account Access Control Proxy).
# MAGIC O papel `roles/tagPolicy.assigner` é concedido apenas às identidades que ainda não têm `assigner` nem `manager`, preservando as regras existentes.

# COMMAND ----------

ACCOUNT_ID = w.api_client.do("GET", "/.well-known/databricks-config")["account_id"]
ASSIGNER, MANAGER = "roles/tagPolicy.assigner", "roles/tagPolicy.manager"
principal_ids = [f"users/{p}" if "@" in p else f"servicePrincipals/{p}" for p in PRINCIPALS]

for key in POLICIES:
    policy = w.tag_policies.get_tag_policy(key)
    name = f"accounts/{ACCOUNT_ID}/tagPolicies/{policy.id}/ruleSets/default"
    rule_set = w.account_access_control_proxy.get_rule_set(name, etag="")
    rules = rule_set.grant_rules or []
    granted = {p for r in rules if r.role in (ASSIGNER, MANAGER) for p in r.principals or []}
    missing = [p for p in principal_ids if p not in granted]
    if missing:
        others = [r for r in rules if r.role != ASSIGNER]
        assigners = [p for r in rules if r.role == ASSIGNER for p in r.principals or []]
        w.account_access_control_proxy.update_rule_set(name, RuleSetUpdateRequest(
            name=name, etag=rule_set.etag, grant_rules=others + [GrantRule(role=ASSIGNER, principals=assigners + missing)]))
        actions[key] += f"; assign {missing}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificação

# COMMAND ----------

for key, (_, values) in POLICIES.items():
    current = [v.name for v in w.tag_policies.get_tag_policy(key).values]
    assert set(values) <= set(current), (key, current)
print(actions)
dbutils.notebook.exit(str(actions))
