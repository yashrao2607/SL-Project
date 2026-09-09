$log = "D:\SL Project\results\local_tier1_log.txt"
while ($true) {
  Start-Sleep -Seconds 60
  if ((Test-Path $log) -and (Select-String -Path $log -Pattern "\[grid\] 800 jobs" -Quiet)) {
    Get-CimInstance Win32_Process -Filter "Name='python3.11.exe'" | Where-Object { $_.CommandLine -like "*03_run_tier1.py*" -or $_.CommandLine -like "*multiprocessing*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $_.CommandLine -like "*run_local_share*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Add-Content -Path "D:\SL Project\results\local_tier1_log.txt" -Value "[watcher] estrogen-alpha finished; redundant stages stopped at $(Get-Date -Format HH:mm:ss)"
    break
  }
}
