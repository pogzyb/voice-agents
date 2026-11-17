#!/usr/bin/env bash
set -euo pipefail

RG="${RG:-rg-voicetools-eastus}"
LC="${LC:-eastus}"
FN="${FN:-fn-voicetools-eastus}"
PYTHON_VERSION="${PYTHON_VERSION:-3.14}"

command -v az >/dev/null
command -v func >/dev/null

SUBSCRIPTION_ID="$(az account show --query id --output tsv)"
SUBSCRIPTION_ID="${SUBSCRIPTION_ID//-/}"
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-voicetools${SUBSCRIPTION_ID:0:12}}"

az group create --name "$RG" --location "$LC"

az storage account create \
	--resource-group "$RG" \
	--name "$STORAGE_ACCOUNT" \
	--location "$LC" \
	--sku Standard_LRS \
	--allow-blob-public-access false

az functionapp create \
	--resource-group "$RG" \
	--name "$FN" \
	--storage-account "$STORAGE_ACCOUNT" \
	--flexconsumption-location "$LC" \
	--runtime python \
	--runtime-version "$PYTHON_VERSION"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
func azure functionapp publish "$FN"

az functionapp show --resource-group "$RG" --name "$FN" --output tsv

# No auth — public MCP server
azd ai connection create mcp-voicetools \
  --kind remote-tool \
  --target "https://fn-voicetools-eastus.azurewebsites.net/runtime/webhooks/mcp" \
  --auth-type none
