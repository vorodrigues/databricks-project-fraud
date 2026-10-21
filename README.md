# fraud_mlops — Bundle MLOps do modelo antifraude PIX

Databricks Asset Bundle (DABs) no layout [MLOps Stacks](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-stacks) que empacota o ciclo de vida do modelo de detecção de fraude em transações PIX — dados, features, treino, registro, batch scoring e online serving — e o promove entre ambientes com GitHub Actions.

## Estrutura

```
databricks.yml                              # bundle, variáveis e targets
resources/
  ml-artifacts-resource.yml                 # experimento MLflow e modelo registrado no Unity Catalog
  setup-workflow-resource.yml               # job fraud_setup
  training-workflow-resource.yml            # job fraud_training
  batch-scoring-workflow-resource.yml       # job fraud_batch_scoring
  online-deploy-workflow-resource.yml       # job fraud_online_deploy
setup/notebooks/                            # data, online_store, tag_policy
training/notebooks/                         # feature_batch, feature_ondemand, train, register
scoring/notebooks/                          # batch_scoring
serving/notebooks/                          # online_deploy
.github/workflows/
  ci.yml                                    # PR → validate, deploy e treino em staging
  cd.yml                                    # merge na main → validate e deploy em prod
.DRAFTS/                                    # testes one-off (fora do git e do deploy)
```

## Targets

Todos os targets usam o mesmo workspace e compute Serverless.

| Target | Modo | Deploy e `run_as` | Schema UC | Diretório raiz |
|---|---|---|---|---|
| `dev` (default) | `development` | usuário desenvolvedor | `workspace.poc_ml_dev` | `/Workspace/Users/<usuário>/mlops/dev` |
| `staging` | `production` | SP `cicd-ml-serving` (via CI) | `workspace.poc_ml_staging` | `/Workspace/mlops/staging` |
| `prod` | `production` | SP `cicd-ml-serving` (via CD) | `workspace.poc_ml` | `/Workspace/mlops/prod` |

Em `development`, o DAB prefixa os nomes dos recursos com o usuário (ex.: experimento `[dev <usuário>] fraud_experiment`, modelo `dev_<usuário>_fraud_model`) e pausa agendamentos, isolando o trabalho de cada desenvolvedor.

## Variáveis

| Variável | Default | Uso |
|---|---|---|
| `catalog` | `workspace` | catálogo UC |
| `schema` | por target | schema UC do target |
| `prefix` | `${bundle.target}` | prefixo de ativos sem diretório (jobs, endpoint) |
| `online_store` | `feature-store-online` | instância Lakebase compartilhada entre targets |
| `service_principal` | application id do SP | `run_as` de `staging`/`prod` |
| `viewers_group` | `ml-viewers` | grupo com permissão de execução/consulta |

O código é implantado em `${workspace.file_path}` (`<diretório raiz>/files`). Os jobs recebem esses valores como job parameters e os notebooks os leem via `dbutils.widgets` — nenhum valor de ambiente é fixado no código.

## Recursos

| Recurso | Nome | Descrição |
|---|---|---|
| Experimento MLflow | `${workspace.root_path}/fraud_experiment` | runs de tuning e treino final |
| Modelo registrado (UC) | `${var.catalog}.${var.schema}.fraud_model` | versões com alias `champion` |
| Job | `${var.prefix}_fraud_setup` | `data`, `online_store`, `tag_policy` (independentes, idempotentes) |
| Job | `${var.prefix}_fraud_training` | `feature_batch` + `feature_ondemand` → `train` → `quality_gate` (`test_auc >= min_auc`) → `register` |
| Job | `${var.prefix}_fraud_batch_scoring` | pontua a tabela fonte com `fraud_model@champion` (diário 06:00 BRT, pausado) |
| Job | `${var.prefix}_fraud_online_deploy` | cria/atualiza o endpoint route-optimized `${var.prefix}_fraud_serving` com inference table |

Todos os jobs concedem `CAN_MANAGE_RUN` ao grupo `ml-viewers`.

## Pré-requisitos

- Service principal `cicd-ml-serving` e grupo `ml-viewers` no workspace.
- Pasta `/Workspace/mlops` com `CAN_MANAGE` para o SP.
- Schemas `workspace.poc_ml_dev`, `workspace.poc_ml_staging` e `workspace.poc_ml`, com privilégios para o usuário e o SP.
- Secret scope `mlops` (`sp-client-id`, `sp-client-secret`) para os testes do endpoint.
- Secrets do repositório GitHub: `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` (OAuth M2M do SP).
- Execução única de `fraud_setup` em cada target antes dos demais jobs.

## Desenvolvimento em `dev`

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev fraud_setup        # uma vez
databricks bundle run -t dev fraud_training
databricks bundle run -t dev fraud_batch_scoring
databricks bundle run -t dev fraud_online_deploy
```

## Fluxo de promoção

1. **dev** — desenvolvimento e testes com o próprio usuário.
2. **staging** — um PR para a `main` dispara o CI (`ci.yml`): `bundle validate`, `bundle deploy` e `bundle run fraud_training` em `staging`, como SP. O workflow aguarda a run e falha se ela falhar.
3. **prod** — o merge na `main` dispara o CD (`cd.yml`): `bundle validate` e `bundle deploy` em `prod`, como SP, sem executar jobs.

Cada workflow usa um grupo de `concurrency` por target, sem cancelar execuções em andamento. Ambos também podem ser disparados manualmente (`workflow_dispatch`). `staging` e `prod` são alterados somente pelo pipeline de CI/CD.
