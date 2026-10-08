# Knowledge drift check, meant to run on a schedule (Windows Task Scheduler).
#   1. kb ingest <Inbox>  - new files dropped in the inbox folder are added; changed ones become new versions
#   2. kb refresh         - every document's recorded source (file path or URL) is re-checked
#   3. kb stale           - documents past their review date, expired, or whose source disappeared
# Output is appended to logs\drift-YYYY-MM-DD.log. Exit code 1 means something needs a person to look.
#
# Run by hand:   powershell -ExecutionPolicy Bypass -File scripts\drift-check.ps1 -Inbox C:\askgov\inbox
param(
    [string]$Inbox
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$kb = Join-Path $root ".venv\Scripts\kb.exe"
New-Item -ItemType Directory -Force (Join-Path $root "logs") | Out-Null
$log = Join-Path $root ("logs\drift-{0:yyyy-MM-dd}.log" -f (Get-Date))

function Run-Step([string[]]$kbArgs) {
    Add-Content $log "`n=== $(Get-Date -Format s)  kb $($kbArgs -join ' ')"
    & $kb @kbArgs *>&1 | ForEach-Object { "$_" } | Add-Content $log
    return $LASTEXITCODE
}

$attention = 0
if ($Inbox) {
    if (Test-Path $Inbox) {
        if ((Run-Step @("ingest", $Inbox)) -ne 0) { $attention = 1 }
    } else {
        Add-Content $log "`n=== $(Get-Date -Format s)  inbox folder not found: $Inbox"
        $attention = 1
    }
}
if ((Run-Step @("refresh")) -ne 0) { $attention = 1 }
if ((Run-Step @("stale")) -ne 0) { $attention = 1 }

Add-Content $log "`n=== $(Get-Date -Format s)  done, needs attention: $([bool]$attention)"
exit $attention
