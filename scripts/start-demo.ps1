<#
Starts PetroKnowledge for a demo and keeps it up (K28): Postgres in WSL, the three
model servers, the web app and the quick tunnel. Every minute it checks each one and
starts whatever is down, so a crash or a reboot of one part does not end the demo.
The tunnel's current link is written to data/link.txt (a quick tunnel gets a new link
each time it starts).

Only this repository's own cloudflared (tools/cloudflared) is started or checked; any
other cloudflared on the machine is left alone.

Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts/start-demo.ps1
Add -Once to start what is down and exit.
#>
param([switch]$Once)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Logs = Join-Path $Root "data\logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$Server = Join-Path $Root "tools\llamacpp\llama-server.exe"
$Tunnel = Join-Path $Root "tools\cloudflared\cloudflared.exe"

# The model servers' lines, as measured for this laptop (Lime Green ADR 0029).
$Models = @(
    @{ Name = "reranker"; Port = 8082; Args = "-m models\bge-reranker-v2-m3-Q8_0.gguf --reranking -np 1 -c 2048 -b 2048 -ub 2048 --cache-ram 0 -ngl all --fit off --host 127.0.0.1 --port 8082" },
    @{ Name = "generator"; Port = 8080; Args = "-m models\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf -c 32768 -np 2 --no-kv-unified -ngl all --n-cpu-moe 99 --fit off --load-mode none -b 2048 -ub 2048 --cache-ram 2048 --host 127.0.0.1 --port 8080" },
    @{ Name = "embedder"; Port = 8084; Args = "-m models\Qwen3-Embedding-4B-GGUF\Qwen3-Embedding-4B-Q8_0.gguf --embedding --pooling last -np 1 -c 2048 -b 2048 -ub 2048 --cache-ram 0 --device none --host 127.0.0.1 --port 8084" },
    # The grounding judge (K28) starts last: llama.cpp fits its layers into the GPU
    # memory the generator leaves and runs the rest on the CPU.
    @{ Name = "judge"; Port = 8085; Args = "-m models\NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf -c 8192 -np 1 --cache-ram 0 --host 127.0.0.1 --port 8085" }
)

function Test-Url($Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-Port($Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $Port)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Write-Event($Text) {
    $line = "{0:u} {1}" -f (Get-Date), $Text
    Write-Output $line
    Add-Content -Path (Join-Path $Logs "start-demo.log") -Value $line -Encoding UTF8
}

function Start-Postgres {
    if (Test-Port 5433) { return }
    Write-Event "postgres down: starting WSL and the dev database"
    # WSL stops its VM soon after the last Windows-side command; a sleeping shell keeps it up.
    Start-Process -WindowStyle Hidden wsl -ArgumentList '-d Ubuntu -- bash -lc "exec sleep infinity"'
    $linuxRoot = (wsl -d Ubuntu -- wslpath -a ($Root -replace '\\', '/')).Trim()
    wsl -d Ubuntu -- bash -lc "cd '$linuxRoot' && docker compose -f deploy/compose.yaml --profile dev up -d" | Out-Null
}

function Start-Models {
    foreach ($model in $Models) {
        if (Test-Url "http://127.0.0.1:$($model.Port)/health") { continue }
        Write-Event "$($model.Name) down: starting on $($model.Port)"
        $log = Join-Path $Logs "$($model.Name).log"
        Start-Process -WindowStyle Hidden -FilePath $Server -ArgumentList $model.Args `
            -RedirectStandardError $log -RedirectStandardOutput "$log.out"
    }
}

function Start-App {
    if (Test-Url "http://127.0.0.1:8091/healthz") { return }
    Write-Event "web app down: starting on 8091"
    $log = Join-Path $Logs "serve.log"
    Start-Process -WindowStyle Hidden -FilePath "uv" `
        -ArgumentList "run --no-sync --env-file .env petroknowledge serve" `
        -RedirectStandardError $log -RedirectStandardOutput "$log.out"
}

function Start-Tunnel {
    $ours = Get-Process cloudflared -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $Tunnel }
    if ($ours) { return }
    Write-Event "tunnel down: starting a quick tunnel"
    $log = Join-Path $Logs "tunnel.log"
    Start-Process -WindowStyle Hidden -FilePath $Tunnel `
        -ArgumentList "tunnel --no-autoupdate --url http://127.0.0.1:8091" `
        -RedirectStandardError $log -RedirectStandardOutput "$log.out"
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        $found = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) {
            $link = $found.Matches[0].Value
            Set-Content -Path (Join-Path $Root "data\link.txt") -Value $link -Encoding UTF8
            Write-Event "link: $link"
            return
        }
    }
    Write-Event "tunnel started, but no link found in $log yet"
}

while ($true) {
    Start-Postgres
    Start-Models
    Start-App
    Start-Tunnel
    if ($Once) { break }
    Start-Sleep -Seconds 60
}
